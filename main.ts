// Sokoban solver: push-optimal A* per room.
//
// A level is a one-square-wide hallway with rooms off it, each room an
// independent subproblem: a box pushed out of a room's door into the hallway
// can never be pushed back, so every box stays in its room and the level's
// push-optimal count is the sum of its rooms' optima. The solver therefore
// cuts the level into rooms, solves each room push-optimally on its own, and
// joins the rooms' move strings with the player's walks between them.
//
// Each room is solved by A* over pushes (a state is the boxes plus the
// lowest square of the player's reachable region, which canonicalises the
// player's position), with:
//   - an admissible heuristic: the minimum-cost assignment of boxes to goals
//     under shortest-path distances over open floor (exact for <= 4 boxes);
//   - deadlock pruning: dead squares (found by pulling a box back from the
//     goals), frozen 2x2 blocks of walls and boxes, and frozen boxes (a box
//     whose every push is blocked by a wall, a dead square or a frozen box).
//
// All budgets are counted in expanded states, never in wall-clock time, and
// every tie is broken by a fixed order, so two runs give byte-identical
// output. Only Node's built-in modules are used; the file runs as
// `node main.ts` on Node 22 (erasable TypeScript only: no enums, no
// parameter properties, no namespaces).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Budgets in expanded states. A room the generator keeps is solvable by a
// plain push BFS within 250,000 states, and A* with the assignment heuristic
// expands far fewer, so these caps are headroom, not the usual path.
const ROOM_ASTAR_BUDGET = 200_000;
const ROOM_BFS_BUDGET = 250_000;
const ROOM_GREEDY_BUDGET = 100_000;
const WHOLE_ASTAR_BUDGET = 200_000;
const WHOLE_BFS_BUDGET = 250_000;

/** The four moves, in a fixed order used for every tie-break: [dx, dy, letter]. */
const DIRS: ReadonlyArray<readonly [number, number, string]> = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Board {
  w: number;
  h: number;
  /** 1 = wall. For a room's sub-board this also holds the other rooms' boxes. */
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  cells: number[];
  boxes: number[];
  goals: number[];
}

interface Push {
  from: number;
  to: number;
  dir: number;
}

interface SearchOutcome {
  moves: string;
  pushes: number;
  expansions: number;
  optimal: boolean;
}

/** Parse XSB text the way the checker does: short rows pad out as walls. */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(0, ...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The squares from which a box can be pushed onto some goal: pull a box back
 * from the goals. A square y is live when a live neighbour x exists such that
 * the square beyond y, away from x, is open floor for the player to stand on.
 */
function liveSquares(w: number, h: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(w * h);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    const xx = x % w;
    const xy = (x / w) | 0;
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k];
      const dx = d[0];
      const dy = d[1];
      const yx = xx - dx;
      const yy = xy - dy;
      if (yx < 0 || yx >= w || yy < 0 || yy >= h) continue;
      const y = yy * w + yx;
      const zx = yx - dx;
      const zy = yy - dy;
      if (zx < 0 || zx >= w || zy < 0 || zy >= h) continue;
      const z = zy * w + zx;
      if (!wall[y] && !wall[z] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Split a level into its rooms: cut the dead squares of the player's row
 * (the hallway: a box there reaches no goal) out of the floor, and each
 * connected part left that holds a box is a room. Null when the level is no
 * such set of rooms (a box in the hallway, a goal cut off, or a room whose
 * boxes and goals do not match), in which case the whole level is solved as
 * one problem instead.
 */
function roomsOf(level: Board, live: Uint8Array): Room[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const size = w * h;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  for (const b of boxes) if (cut[b]) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const cells: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      const cx = c % w;
      const cy = (c / w) | 0;
      if (cx > 0) {
        const q = c - 1;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          cells.push(q);
        }
      }
      if (cx < w - 1) {
        const q = c + 1;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          cells.push(q);
        }
      }
      if (cy > 0) {
        const q = c - w;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          cells.push(q);
        }
      }
      if (cy < h - 1) {
        const q = c + w;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          cells.push(q);
        }
      }
    }
    rooms.push({ cells, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The player's square after playing `moves` (LURD, either case) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    if (ch === 'u' || ch === 'U') p -= w;
    else if (ch === 'd' || ch === 'D') p += w;
    else if (ch === 'l' || ch === 'L') p -= 1;
    else if (ch === 'r' || ch === 'R') p += 1;
  }
  return p;
}

/** The shortest walk from `from` to `to` around walls and the boxes in `occupied`. */
function walkTo(board: Board, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const { w, h, wall } = board;
  const size = w * h;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    const px = p % w;
    const py = (p / w) | 0;
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k];
      const nx = px + d[0];
      const ny = py + d[1];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const q = ny * w + nx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d[2];
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** Turn a push list into a move string, walking the player to each push first. */
function movesFromPushes(board: Board, pushes: Push[]): string {
  const { w, h, wall, boxes: startBoxes, player: startPlayer } = board;
  const occupied = new Uint8Array(w * h);
  for (const b of startBoxes) occupied[b] = 1;
  const boxes = startBoxes.slice();
  let player = startPlayer;
  let moves = '';
  for (const p of pushes) {
    const d = DIRS[p.dir];
    const stand = p.from - (d[1] * w + d[0]);
    moves += walkTo(board, occupied, player, stand);
    moves += d[2].toUpperCase();
    occupied[p.from] = 0;
    occupied[p.to] = 1;
    const at = boxes.indexOf(p.from);
    boxes[at] = p.to;
    boxes.sort((a, b) => a - b);
    player = p.from;
  }
  return moves;
}

type SearchMode = 'astar' | 'bfs' | 'greedy';

/**
 * Push-optimal search over one board (a room's sub-board, or a whole level):
 * A* by default, plain push BFS, or greedy best-first as a last resort.
 * Returns null when no solution was found inside the budget.
 */
function searchPushes(board: Board, maxExpansions: number, mode: SearchMode): SearchOutcome | null {
  const { w, h, wall, goals, boxes: startBoxes, player: startPlayer } = board;
  const size = w * h;
  const k = startBoxes.length;
  if (k === 0) return { moves: '', pushes: 0, expansions: 0, optimal: true };
  if (startPlayer < 0 || goals.length === 0) return null;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(w, h, wall, goals);
  for (const b of startBoxes) if (!live[b]) return null;

  // Shortest-path distances from every square to each goal, boxes ignored:
  // a box needs at least this many pushes to reach a goal.
  const distToGoal: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(size).fill(-1);
    const queue: number[] = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi];
      const cx = c % w;
      const cy = (c / w) | 0;
      if (cx > 0) {
        const n = c - 1;
        if (!wall[n] && dist[n] < 0) {
          dist[n] = dist[c] + 1;
          queue.push(n);
        }
      }
      if (cx < w - 1) {
        const n = c + 1;
        if (!wall[n] && dist[n] < 0) {
          dist[n] = dist[c] + 1;
          queue.push(n);
        }
      }
      if (cy > 0) {
        const n = c - w;
        if (!wall[n] && dist[n] < 0) {
          dist[n] = dist[c] + 1;
          queue.push(n);
        }
      }
      if (cy < h - 1) {
        const n = c + w;
        if (!wall[n] && dist[n] < 0) {
          dist[n] = dist[c] + 1;
          queue.push(n);
        }
      }
    }
    distToGoal.push(dist);
  }

  // All orders of the goals, for the exact minimum-cost assignment (<= 4 boxes).
  const perms: number[][] = [];
  if (k <= 4) {
    const base: number[] = [];
    for (let i = 0; i < k; i++) base.push(i);
    const gen = (arr: number[], n: number): void => {
      if (n === 1) {
        perms.push(arr.slice());
        return;
      }
      for (let i = 0; i < n; i++) {
        gen(arr, n - 1);
        const j = n % 2 === 1 ? 0 : i;
        const t = arr[j];
        arr[j] = arr[n - 1];
        arr[n - 1] = t;
      }
    };
    gen(base, k);
  }
  const hCache = new Map<string, number>();
  const heuristic = (boxes: number[]): number => {
    const key = boxes.join(',');
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let best = Infinity;
    if (k <= 4) {
      for (const perm of perms) {
        let s = 0;
        for (let i = 0; i < k; i++) {
          const d = distToGoal[perm[i]][boxes[i]];
          if (d < 0) {
            s = Infinity;
            break;
          }
          s += d;
        }
        if (s < best) best = s;
      }
    } else {
      // More boxes than the exact assignment covers: each box to its nearest goal.
      let s = 0;
      for (let i = 0; i < k; i++) {
        let m = Infinity;
        for (let gi = 0; gi < goals.length; gi++) {
          const d = distToGoal[gi][boxes[i]];
          if (d >= 0 && d < m) m = d;
        }
        if (m === Infinity) {
          s = Infinity;
          break;
        }
        s += m;
      }
      best = s;
    }
    hCache.set(key, best);
    return best;
  };

  // Working scratch: two stamp arrays for the reachable regions (the expanding
  // state's, and a successor's), one stack, and the box occupancy.
  const seen = new Int32Array(size);
  const seen2 = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stamp = 0;

  /** Lowest square of the region reachable from `from`; marks `mark` with `st`. */
  const reachLow = (from: number, mark: Int32Array, st: number): number => {
    let low = from;
    let n = 0;
    stack[n++] = from;
    mark[from] = st;
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      const px = p % w;
      const py = (p / w) | 0;
      if (px > 0) {
        const q = p - 1;
        if (!wall[q] && !occupied[q] && mark[q] !== st) {
          mark[q] = st;
          stack[n++] = q;
        }
      }
      if (px < w - 1) {
        const q = p + 1;
        if (!wall[q] && !occupied[q] && mark[q] !== st) {
          mark[q] = st;
          stack[n++] = q;
        }
      }
      if (py > 0) {
        const q = p - w;
        if (!wall[q] && !occupied[q] && mark[q] !== st) {
          mark[q] = st;
          stack[n++] = q;
        }
      }
      if (py < h - 1) {
        const q = p + w;
        if (!wall[q] && !occupied[q] && mark[q] !== st) {
          mark[q] = st;
          stack[n++] = q;
        }
      }
    }
    return low;
  };

  /** Whether the box at `cell` is frozen, given the boxes and their frozen flags. */
  const frozenBoxAt = (cell: number, boxes: number[], frozen: Uint8Array): boolean => {
    for (let i = 0; i < boxes.length; i++) if (boxes[i] === cell) return frozen[i] === 1;
    return false;
  };

  /**
   * Whether a push of the box at `b` in direction (dx, dy) could ever happen:
   * the destination is open floor a box can be pushed onto, and the square
   * the player stands on is open floor; a square holding a box that is not
   * (yet) frozen does not block it, since that box may move away first.
   */
  const dirOpen = (b: number, dx: number, dy: number, boxes: number[], frozen: Uint8Array): boolean => {
    const bx = b % w;
    const by = (b / w) | 0;
    const nx = bx + dx;
    const ny = by + dy;
    if (nx < 0 || nx >= w || ny < 0 || ny >= h) return false;
    const dest = ny * w + nx;
    const sx = bx - dx;
    const sy = by - dy;
    if (sx < 0 || sx >= w || sy < 0 || sy >= h) return false;
    const stand = sy * w + sx;
    if (wall[dest] || !live[dest]) return false;
    if (occupied[dest] && frozenBoxAt(dest, boxes, frozen)) return false;
    if (wall[stand]) return false;
    if (occupied[stand] && frozenBoxAt(stand, boxes, frozen)) return false;
    return true;
  };

  /**
   * True when some box off its goal can never be pushed again: no direction
   * is open for it, where a direction is blocked by a wall, a dead square,
   * or a box already known frozen. Computed to a fixpoint; a box on a goal
   * counts as frozen for blocking others, but never triggers the prune.
   */
  const hasFrozenBox = (boxes: number[]): boolean => {
    const n = boxes.length;
    const frozen = new Uint8Array(n);
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < n; i++) {
        if (frozen[i]) continue;
        let movable = false;
        for (let k2 = 0; k2 < 4; k2++) {
          const d = DIRS[k2];
          if (dirOpen(boxes[i], d[0], d[1], boxes, frozen)) {
            movable = true;
            break;
          }
        }
        if (!movable) {
          frozen[i] = 1;
          changed = true;
        }
      }
    }
    for (let i = 0; i < n; i++) if (frozen[i] && !goalSet[boxes[i]]) return true;
    return false;
  };

  /** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes. */
  const frozenBlockAt = (t: number): boolean => {
    const tx = t % w;
    const ty = (t / w) | 0;
    const corners: Array<[number, number]> = [
      [tx, ty],
      [tx - 1, ty],
      [tx, ty - 1],
      [tx - 1, ty - 1],
    ];
    for (const [ax, ay] of corners) {
      if (ax < 0 || ay < 0 || ax + 1 >= w || ay + 1 >= h) continue;
      const a = ay * w + ax;
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if ((wall[a] || occupied[a]) && (wall[b] || occupied[b]) && (wall[c] || occupied[c]) && (wall[d] || occupied[d])) {
        if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) {
          return true;
        }
      }
    }
    return false;
  };

  // A state's key: its boxes and the lowest square of the player's region.
  // With at most 4 boxes on a board of at most 256 squares this packs into
  // one exact number; otherwise it is a string.
  const useNumericKey = k <= 4 && size <= 255;
  const stateKey = (boxes: number[], low: number): number | string => {
    if (!useNumericKey) return `${low}|${boxes.join(',')}`;
    let key = low;
    for (let i = 0; i < 4; i++) key = key * 256 + (i < boxes.length ? boxes[i] : 255);
    return key;
  };

  const isGoal = (boxes: number[]): boolean => {
    for (const b of boxes) if (!goalSet[b]) return false;
    return true;
  };
  if (isGoal(startBoxes)) return { moves: '', pushes: 0, expansions: 0, optimal: true };

  const fOf = (g: number, hh: number): number => (mode === 'astar' ? g + hh : mode === 'bfs' ? g : hh);

  // The states: their boxes, the region's lowest square, pushes so far, and
  // the parent and push that made them. The best map holds, per state key,
  // the stored state with the fewest pushes; heap entries are [f, g, seq, state].
  const stBoxes: number[][] = [];
  const stLow: number[] = [];
  const stG: number[] = [];
  const stParent: number[] = [];
  const stFrom: number[] = [];
  const stTo: number[] = [];
  const stDir: number[] = [];
  const best = new Map<number | string, number>();
  let seq = 0;
  const heap: number[][] = [];
  const heapLess = (a: number[], b: number[]): boolean => {
    if (a[0] !== b[0]) return a[0] < b[0];
    if (a[1] !== b[1]) return a[1] > b[1];
    return a[2] < b[2];
  };
  const pushHeap = (f: number, g: number, s: number): void => {
    const e = [f, g, seq++, s];
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      const t = heap[i];
      heap[i] = heap[p];
      heap[p] = t;
      i = p;
    }
  };
  const popHeap = (): number[] => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0 && last !== undefined) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heapLess(heap[l], heap[m])) m = l;
        if (r < heap.length && heapLess(heap[r], heap[m])) m = r;
        if (m === i) break;
        const t = heap[i];
        heap[i] = heap[m];
        heap[m] = t;
        i = m;
      }
    }
    return top;
  };

  for (const b of startBoxes) occupied[b] = 1;
  const low0 = reachLow(startPlayer, seen, ++stamp);
  for (const b of startBoxes) occupied[b] = 0;
  const h0 = heuristic(startBoxes);
  if (h0 === Infinity) return null;
  stBoxes.push(startBoxes.slice());
  stLow.push(low0);
  stG.push(0);
  stParent.push(-1);
  stFrom.push(-1);
  stTo.push(-1);
  stDir.push(-1);
  best.set(stateKey(startBoxes, low0), 0);
  pushHeap(fOf(0, h0), 0, 0);

  let expansions = 0;
  let bestGoal = -1;
  let bestGoalG = Infinity;

  const finish = (goalState: number, optimal: boolean): SearchOutcome => {
    const pushes: Push[] = [];
    for (let s = goalState; s > 0; s = stParent[s]) {
      pushes.push({ from: stFrom[s], to: stTo[s], dir: stDir[s] });
    }
    pushes.reverse();
    return { moves: movesFromPushes(board, pushes), pushes: pushes.length, expansions, optimal };
  };

  while (heap.length > 0) {
    const entry = popHeap();
    const s = entry[3];
    const g = stG[s];
    if (best.get(stateKey(stBoxes[s], stLow[s])) !== s) continue;
    if (isGoal(stBoxes[s])) return finish(s, mode !== 'greedy');
    if (expansions >= maxExpansions) break;
    expansions++;
    const boxes = stBoxes[s];
    for (const b of boxes) occupied[b] = 1;
    const myStamp = ++stamp;
    reachLow(stLow[s], seen, myStamp);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      const bx = b % w;
      const by = (b / w) | 0;
      for (let k2 = 0; k2 < 4; k2++) {
        const d = DIRS[k2];
        const dx = d[0];
        const dy = d[1];
        const nx = bx + dx;
        const ny = by + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const t = ny * w + nx;
        const sx = bx - dx;
        const sy = by - dy;
        if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
        const stand = sy * w + sx;
        if (wall[t] || occupied[t] || !live[t]) continue;
        if (seen[stand] !== myStamp) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlockAt(t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        if (hasFrozenBox(next)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const hh = heuristic(next);
        if (hh === Infinity) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const low = reachLow(b, seen2, ++stamp);
        const key = stateKey(next, low);
        const prev = best.get(key);
        const ng = g + 1;
        if (prev !== undefined && stG[prev] <= ng) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const ns = stBoxes.length;
        stBoxes.push(next);
        stLow.push(low);
        stG.push(ng);
        stParent.push(s);
        stFrom.push(b);
        stTo.push(t);
        stDir.push(k2);
        best.set(key, ns);
        if (isGoal(next) && ng < bestGoalG) {
          bestGoal = ns;
          bestGoalG = ng;
        }
        pushHeap(fOf(ng, hh), ng, ns);
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (bestGoal >= 0) return finish(bestGoal, false);
  return null;
}

/** A* first, then push BFS, then greedy: the first search that finds a solution wins. */
function searchRoom(board: Board, astarBudget: number, bfsBudget: number, greedyBudget: number): SearchOutcome | null {
  const a = searchPushes(board, astarBudget, 'astar');
  if (a !== null) return a;
  const b = searchPushes(board, bfsBudget, 'bfs');
  if (b !== null) return b;
  return searchPushes(board, greedyBudget, 'greedy');
}

/** Solve one level: room by room, or whole when it is no set of rooms. */
function solveLevel(level: Board): string {
  const live = liveSquares(level.w, level.h, level.wall, level.goals);
  const rooms = roomsOf(level, live);
  if (rooms === null) {
    const whole = searchRoom(level, WHOLE_ASTAR_BUDGET, WHOLE_BFS_BUDGET, ROOM_GREEDY_BUDGET);
    return whole ? whole.moves : '';
  }
  let moves = '';
  let player = level.player;
  const done: boolean[] = rooms.map(() => false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const wall = level.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const other = rooms[oj];
      const spots = done[oj] ? other.goals : other.boxes;
      for (const b of spots) wall[b] = 1;
    }
    const sub: Board = { w: level.w, h: level.h, wall, goals: room.goals.slice(), boxes: room.boxes.slice(), player };
    const result = searchRoom(sub, ROOM_ASTAR_BUDGET, ROOM_BFS_BUDGET, ROOM_GREEDY_BUDGET);
    if (result === null) {
      const whole = searchRoom(level, WHOLE_ASTAR_BUDGET, WHOLE_BFS_BUDGET, ROOM_GREEDY_BUDGET);
      return whole ? whole.moves : '';
    }
    moves += result.moves;
    player = endOf(level.w, player, result.moves);
    done[ri] = true;
  }
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
    const level = parseLevel(text);
    if (level.player >= 0 && level.boxes.length > 0) moves = solveLevel(level);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
