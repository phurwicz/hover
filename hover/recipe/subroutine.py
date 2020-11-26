"""
Functions commonly used by classes in this submodule.

Note that functions which are also used outside this submodule should be moved up.
"""


def link_size_and_range(*figures, height=600, width=800):
    """
    Give the specified figures the same size and display range.

    This also synchronizes zooming in or out.
    """
    for i, _fi in enumerate(figures):
        # set plots to the same size
        _fi.plot_height = height
        _fi.plot_width = width
        for j, _fj in enumerate(figures):
            if not i == j:
                # link coordinate ranges
                for _attr in ["start", "end"]:
                    _fi.x_range.js_link(_attr, _fj.x_range, _attr)
                    _fi.y_range.js_link(_attr, _fj.y_range, _attr)


def link_selection(*sources):
    """
    Sync the selected indices between multiple sources.
    """
    for i, _si in enumerate(sources):
        for j, _sj in enumerate(sources):
            if not i == j:
                # link selection
                _si.selected.js_link("indices", _sj.selected, "indices")
