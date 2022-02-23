> `hover` creates a [`bokeh` server app](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) to deliver its annotation interface.
>
> :rocket: This app can be served flexibly based on your needs.

{!docs/snippets/html/stylesheet.html!}

## **Prerequisites**

Suppose that we've already used a `recipe` to create a `handle` function like in the [quickstart](../t0-quickstart/#apply-labels).

??? info "Recap from the tutorials before"
    -   the `handle` is a function which renders plot elements on a [`bokeh` document](https://docs.bokeh.org/en/latest/docs/reference/document.html).

## **Option 1: Jupyter**

We are probably familiar with this now:

```Python
from bokeh.io import show, output_notebook
output_notebook()
show(handle) # notebook_url='http://localhost:8888'
```

???+ tip "Pros & Cons"
    This inline Jupyter mode can integrate particularly well with your notebook workflow. For example, when your are (tentatively) done with annotation, the `SupervisableDataset` can be accessed directly in the notebook, rather than exported to a file and loaded back.

    The inline mode is highly recommended for local usage.

    -   On the contrary, with a remote Jupyter server, it may have trouble displaying the plots.

        -   this can be due to failure of loading JavaScript libraries or accessing implicit bokeh server ports.

## **Option 2: Command Line**

[`bokeh serve`](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) starts an explicit `tornado` server from the command line:

```bash
bokeh serve my-app.py
```

```Python
# my-app.py

# handle = ...

from bokeh.io import curdoc
doc = curdoc()
handle(doc)
```

???+ tip "Pros & Cons"
    This is the "classic" approach to run a `bokeh` server. Remote access is simple through parameters [**specified here**](https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html). The bokeh plot tools are mobile-friendly too -- this means you can host a server, e.g. an http-enabled cloud virtual machine, and annotate from a tablet.

    The command line mode is less interactive, since Python objects in the script cannot be accessed on the fly.

## **Option 3: Anywhere in Python**

It is possible to [embed the app](https://docs.bokeh.org/en/latest/docs/user_guide/server.html#embedding-bokeh-server-as-a-library) in regular Python:

```Python
from bokeh.server.server import Server
server = Server({'/my-app': handle})
server.start()
```

???+ tip "Pros & Cons"
    This embedded mode is a go-to for serving within a greater application.

    Also note that each command line argument for `bokeh serve` has a corresponding keyword argument to `Server()`.

    For instance, `bokeh serve <args> --allow-websocket-origin=*` in the command line mirrors `Server(*args, allow_websocket_origin='*')` in Python.

    The embedded mode gives you the most control of your server.
