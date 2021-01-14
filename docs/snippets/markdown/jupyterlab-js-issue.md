??? warning "Known issue"
    {== If you are running this code block on this documentation page: ==}

    -   JavaScript output (which contains the visualization) will fail to render due to JupyterLab's security restrictions.
    -   please run this tutorial locally to view the output.

    ??? help "Advanced: help wanted"
        Some context:

        -   the code blocks here are embedded using [Juniper](https://github.com/ines/juniper).
        -   the environment is configured in the [Binder repo](https://github.com/phurwicz/hover-binder).

        What we've tried:

        -   1 [Bokeh's extension with JupyterLab](https://github.com/bokeh/jupyter_bokeh)
            -   1.1 cannot render the Bokeh plots remotely with `show(handle)`, with or without the extension
                -   1.1.1 JavaScript console suggests that `bokeh.main.js` would fail to load.
        -   2 [JavaScript magic cell](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-javascript)
            -   2.1 such magic is functional in a custom notebook on the Jupyter server.
            -   2.2 such magic is blocked by JupyterLab if ran on the documentation page.

        Tentative clues:

        -   2.1 & 2.2 suggests that somehow JupyterLab behaves differently between Binder itself and Juniper.
        -   Juniper by default [trusts the cells](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-javascript).
        -   making Javascript magic work on this documentation page would be a great step.
