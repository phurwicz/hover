> `hover` can be customized through its module config.
>
> :bulb: Let's explore a few use cases.

{!docs/snippets/markdown/tutorial-required.md!}
{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}

## **Color Palette for Labeled Data Points**

You may want to customize the color palette for better contrast or accessibility, which can depend on specific scenarios.

The snippet below shows an example of default colors assigned to 6 classes. `hover` by default samples [`Turbo256`](https://docs.bokeh.org/en/latest/docs/reference/palettes.html#large-palettes) to accommodate a large number of classes while keeping good contrast.

<pre data-executable>
{!docs/snippets/py/g2-0-color-palette.txt!}
</pre>

You can change the palette using any `bokeh` palette, or any iterable of hex colors like `"#000000"`.
<pre data-executable>
{!docs/snippets/py/g2-1-configure-palette.txt!}
</pre>

???+ note "Config changes should happen early"
    `hover.config` assignments need to happen before plotting your data.

    -   This is because `hover` locks config values for consistency as soon as each config value is read by other code.
    -   Ideally you should change config immediately after `import hover`.

## **Color of Unlabeled Data Points**

For unlabeled data points, `hover` uses a light gray color `"#dcdcdc"`. This is not configured in the color palette above, but here:

<pre data-executable>
{!docs/snippets/py/g2-2-configure-abstain-color.txt!}
</pre>

## **Dimensionality Reduction Method**

`hover` uses dimensionality reduction in a lot of places. It can be cumbersome to find these places and use your preferred method. In such cases a module-level override can be handy:

<pre data-executable>
{!docs/snippets/py/g2-3-configure-reduction-method.txt!}
</pre>

## **Browse more configs**

There are more configurations that are more niche which we will skip here. You can find a full list of configurations, default values, and hints here:

<pre data-executable>
{!docs/snippets/py/g2-4-config-hint.txt!}
</pre>

Happy customizing!

{!docs/snippets/html/stylesheet.html!}
