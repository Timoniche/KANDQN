from collections import namedtuple

Transition = namedtuple(
    typename='Transition',
    field_names=(
        'state',
        'action',
        'next_state',
        'reward'
    )
)
