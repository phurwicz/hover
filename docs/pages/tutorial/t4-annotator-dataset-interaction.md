> `Annotator` provides a map of your data colored by labels.
>
> :speedboat: Let's walk through its visual components and how they interact with the `SupervisableDataset`.

{!docs/snippets/html/thebe.html!}
{!docs/snippets/markdown/binder-kernel.md!}

## **Scatter Plot: Semantically Similar Points are Close Together**

### **hover works by selecting groups of homogeneous points**
Embeddings are often helpful but not perfect. This is why we have tooltips that show the detail of each point on mouse hover.

## **Data Subsets Toggle**
Showing labeled subsets can tell you which parts of the data has been explored and which ones have not. However, you can also turn off the display to focus on the `raw` subset.

## **Selection Tools: Tap, Polygon, Lasso**

### **cumulative selection toggle**

## **Text Search Widget: Include/Exclude**
Keywords or regular expressions can be great starting points for identifying a cluster of similar points based on domain expertise.

### **Preview: Use Search for Selection in Finder**

## **The Plot and The Dataset**
The plotted data points are not always in sync with the underlying dataset, which is a design choice for performance. Instead we will use a trigger for this synchronization.

### **PUSH: synchronize from dataset to plots**


{!docs/snippets/html/stylesheet.html!}
