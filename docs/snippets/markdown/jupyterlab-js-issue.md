??? info "Showcase widgets here are not interactive"
    {== Plotted widgets **on this page** are not interactive, but only for illustration. ==}

    Widgets {== will be interactive when you actually use them ==} (in your local environment or server apps like in the quickstart).

    -   be sure to use a whole `recipe` rather than individual widgets.
    -   if you really want to plot interactive widgets on their own, try `from hover.utils.bokeh_helper import show_as_interactive as show` instead of `from bokeh.io import show`.
        -   this works in your own environment but still not on the documentation page.
        -   [`show_as_interactive`](/hover/pages/reference/utils-bokeh_helper/#hover.utils.bokeh_helper.show_as_interactive) is a simple tweak of `bokeh.io.show` by turning standalone LayoutDOM to an application.
