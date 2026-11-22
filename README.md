# Sokoban solver

Push-optimal Sokoban solver for make.fun's sokoban-solver-01-run-20261121.

A level is a one-square-wide hallway with rooms off it, and a box pushed
out of a room can never come back, so each room is an independent
subproblem. The solver cuts the level into rooms (dead squares of the
hallway row separate them), solves each room push-optimally with A* over
pushes — minimum-cost box-to-goal assignment heuristic, dead-square,
frozen-2x2 and frozen-box pruning, transposition table on (boxes, player's
reachable region) — and joins the rooms' solutions with the player's walks
between them.

Run: `MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<out> node main.ts`
(one run answers every instance; budgets are counted in expanded states,
never in wall-clock time, so runs are byte-identical).
