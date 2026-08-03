from __future__ import annotations

from cluster.generate_f02_nbody import generation_tasks


def test_generation_tasks_are_replica_major_and_dimension_complete() -> None:
    tasks = generation_tasks([0, 101], [2, 4, 10], n_dims=3)

    assert [
        (task.task_index, task.replica, task.n_particles, task.n_dims)
        for task in tasks
    ] == [
        (0, 0, 2, 3),
        (1, 0, 4, 3),
        (2, 0, 10, 3),
        (3, 101, 2, 3),
        (4, 101, 4, 3),
        (5, 101, 10, 3),
    ]


def test_generation_tasks_reject_duplicates_and_invalid_values() -> None:
    for replicas, particles, n_dims in (
        ([0, 0], [2], 3),
        ([0], [2, 2], 3),
        ([-1], [2], 3),
        ([0], [1], 3),
        ([0], [2], 0),
    ):
        try:
            generation_tasks(replicas, particles, n_dims=n_dims)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation grid was accepted")
