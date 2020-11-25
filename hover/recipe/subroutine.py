"""
Functions commonly used by classes in this submodule.

Note that functions which are also used outside this submodule should be moved up.
"""


def link_plots(*explorers, height=600, width=800):
    linked_plots = [*explorers]

    for i, _pi in enumerate(linked_plots):
        # set plots to the same size
        _pi.figure.plot_height = height
        _pi.figure.plot_width = width
        for j, _pj in enumerate(linked_plots):
            if not i == j:
                # link coordinate ranges
                for _attr in ["start", "end"]:
                    _pi.figure.x_range.js_link(_attr, _pj.figure.x_range, _attr)
                    _pi.figure.y_range.js_link(_attr, _pj.figure.y_range, _attr)
                # link selection
                _pi.source.selected.js_link("indices", _pj.source.selected, "indices")

    return linked_plots
