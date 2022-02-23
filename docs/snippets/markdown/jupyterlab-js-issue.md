???+ info "Doc-page limitation"
    {== Plotted widgets on this page are static and for illustration. ==}

    -   In a `hover` recipe in normal environments like your local notebook, widgets will be fully functional.
    -   If you want to plot interactive widgets on their own, try `from hover.utils.bokeh_helper import show_as_interactive as show` instead of `from bokeh.io import show`.
        -   works for your own environment but not for this documentation page.
        -   [`show_as_interactive`](/hover/pages/reference/utils-bokeh_helper/#hover.utils.bokeh_helper.show_as_interactive) is a simple tweak of `bokeh.io.show` by turning standalone LayoutDOM to an application.
