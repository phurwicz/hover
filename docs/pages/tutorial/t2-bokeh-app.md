> `hover` uses a [`bokeh` server app](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) to deliver its annotation interface.
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
show(handle)
```

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

## **Option 3: Anywhere in Python**

It is also possible to [embedded an app](https://docs.bokeh.org/en/latest/docs/user_guide/server.html#embedding-bokeh-server-as-a-library) into regular Python:

```Python
from bokeh.server.server import Server
server = Server({'/my-app': handle})
server.start()
```
