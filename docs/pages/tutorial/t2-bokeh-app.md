> `hover` creates a [`bokeh` server app](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) to deliver its annotation interface.
>
> :rocket: Let's go over a few ways to run this app.

{!docs/snippets/html/stylesheet.html!}

## **Prerequisites**

Suppose that we've already created a `handle` like in the [quickstart](../t0-quickstart/#apply-labels).

This is our app which can be placed flexibly.

??? info "Extended resources"
    -   the `handle` is a function which renders plot elements on a [`bokeh` document](https://docs.bokeh.org/en/latest/docs/reference/document.html).

## **Option 1: Jupyter**

We've seen this in the tutorials before:

```Python
from bokeh.io import show, output_notebook
output_notebook()
show(handle) # notebook_url='http://localhost:8888'
```

???+ tip "Pros & Cons"
    This inline Jupyter mode can integrate particularly well with your workflow. For example, when we are done with the annotation interface, the `SupervisableDataset` can be accessed directly in the notebook, rather than exported to a file and loaded back.

    The inline mode works like a charm locally, but can have trouble loading JS libraries or finding implicit `bokeh server` ports with a remote Jupyter server.

## **Option 2: Command Line**

[`bokeh serve`](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) starts an explicit `Tornado` server from the command line:

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
    This is the classical approach to run a `bokeh` server. Remote access is simple through parameters [**specified here**](https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html). The bokeh plot tools are mobile-friendly too -- this means you can host a server, such as a cloud virtual machine, and annotate from a tablet.

    The command line mode is less interactive as Python objects in the server do not persist like in Jupyter.

## **Option 3: Anywhere in Python**

It is also possible to [embedded an app](https://docs.bokeh.org/en/latest/docs/user_guide/server.html#embedding-bokeh-server-as-a-library) into regular Python:

```Python
from bokeh.server.server import Server
server = Server({'/my-app': handle})
server.start()
```

???+ tip "Pros & Cons"
    This embedded mode is a go-to for serving within a greater application. The various parameters in the command line mode are also available here, per `bokeh`'s documentation:

    > Also note that most every command line argument for bokeh serve has a corresponding keyword argument to Server.
    >
    > For instance, setting the `--allow-websocket-origin` command line argument is equivalent to passing `allow_websocket_origin` as a parameter.

    The embedded mode can seem a bit sophisticated, but is well justified by its flexibility.
