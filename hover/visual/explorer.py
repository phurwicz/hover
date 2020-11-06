import numpy as np
import bokeh
from bokeh.models import CustomJS, ColumnDataSource
from abc import ABC, abstractmethod
from copy import deepcopy
from hover.config import ABSTAIN_ENCODED, ABSTAIN_DECODED, ENCODED_LABEL_KEY
from .config import bokeh_hover_tooltip

class BokehForLabeledText(ABC):
    """
    Base class that keeps template figure settings.
    """

    def __init__(self, **kwargs):
        from bokeh.plotting import figure

        preset_kwargs = {
            "tooltips": bokeh_hover_tooltip(label=True, text=True),
            "output_backend": "webgl",
        }
        preset_kwargs.update(kwargs)
        self.figure = figure(**preset_kwargs)
        self.widgets = []
        self.setup_widgets()
        
    def setup_widgets(self):
        """
        Prepare widgets for interactive functionality.
        """
        from bokeh.models import TextInput
        self.search_pos = TextInput(title="Text contains (enter plain text, or /pattern/flag for regex)")
        self.search_neg = TextInput(title="Text does not contain (enter plain text, or /pattern/flag for regex)")
        self.widgets.extend([self.search_pos, self.search_neg]) 
        
    def view(self):
        """
        Return a formatted figure for display.
        """
        from bokeh.layouts import column
        
        layout = column(
            *self.widgets,
            self.figure
        )
        return layout

    def activate_search(self, source, kwargs, altered_param=('size', 8, 3, 5)):
        """
        Enables string/regex search-and-highlight mechanism.
        Modifies plotting source and kwargs in-place.
        """
        assert isinstance(source, ColumnDataSource)
        assert isinstance(kwargs, dict)
        param_key, param_pos, param_neg, param_default = altered_param
        num_points = len(source.data['text'])
        default_param_list = [kwargs.get(param_key, param_default)] * num_points
        source.add(default_param_list, f'{param_key}')
        kwargs[param_key] = param_key
        
        search_callback = CustomJS(
            args={
                "source": source,
                "key_pos": self.search_pos,
                "key_neg": self.search_neg,
                "param_pos": param_pos,
                "param_neg": param_neg,
                "param_default": param_default,
            },
            code=f"""
            const data = source.data;
            const text = data['text'];
            var arr = data['{param_key}'];
            """ + """
            var search_pos = key_pos.value;
            var search_neg = key_neg.value;
            var valid_pos = (search_pos.length > 0);
            var valid_neg = (search_neg.length > 0);
            
            function determineAttr(candidate)
            {
                var score = 0;
                if (valid_pos) {
                    if (candidate.search(search_pos) >= 0) {
                        score += 1;
                    } else {
                        score -= 2;
                    }
                };
                if (valid_neg) {
                    if (candidate.search(search_neg) < 0) {
                        score += 1;
                    } else {
                        score -= 2;
                    }
                };
                if (score > 0) {
                    return param_pos;
                } else if (score < 0) {
                    return param_neg;
                } else {return param_default;}
            }
            
            function toRegex(search_key) {
                var match = search_key.match(new RegExp('^/(.*?)/([gimy]*)$'));
                if (match) {
                    return new RegExp(match[1], match[2]);
                } else {
                    return search_key;
                }
            }
            
            if (valid_pos) {search_pos = toRegex(search_pos);}
            if (valid_neg) {search_neg = toRegex(search_neg);}
            for (var i = 0; i < arr.length; i++) {
                arr[i] = determineAttr(text[i]);
            }
            
            source.change.emit()
            """
        )
        
        self.search_pos.js_on_change("value", search_callback)
        self.search_neg.js_on_change("value", search_callback)
        
    @abstractmethod
    def plot(self, *args, **kwargs):
        """
        Plot something onto the figure.
        """
        pass


class BokehCorpusExplorer(BokehForLabeledText):
    """
    Plot unlabeled, 2-D-vectorized text data points in a corpus.
    """

    def __init__(self, df_train, **kwargs):
        super().__init__(**kwargs)

        # prepare plot-ready dataframe for train set
        for _key in ["text", "x", "y"]:
            assert _key in df_train.columns
        self.df_train = df_train.copy()

        # plot the train set as a background
        background_kwargs = {
            "color": "gainsboro",
            "line_alpha": 0.3,
            "legend_label": "unlabeled",
        }
        source = ColumnDataSource(self.df_train)
        self._activate_search_on_corpus(source, background_kwargs)
        
        self.figure.circle(
            "x", "y", source=source, **background_kwargs
        )

    def _activate_search_on_corpus(self, source, background_kwargs):
        """
        Assuming that there will not be a labeled dev set.
        """
        self.activate_search(source, background_kwargs, altered_param=('size', 8, 3, 5))
        self.activate_search(source, background_kwargs, altered_param=('fill_alpha', 0.6, 0.4, 0.5))
        self.activate_search(source, background_kwargs, altered_param=('color', 'coral', 'linen', 'gainsboro'))

    def plot(self, *args, **kwargs):
        """
        Does nothing.
        """
        pass

class BokehMarginExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with two versions of labels.
    Could be useful for A/B tests.
    Currently not considering multi-label scenarios.
    """

    def __init__(self, df_train, label_col_a, label_col_b, **kwargs):
        super().__init__(df_train, **kwargs)

        for _key in ["text", label_col_a, label_col_b, "x", "y"]:
            assert _key in df_train.columns
        self.label_col_a = label_col_a
        self.label_col_b = label_col_b

    def _activate_search_on_corpus(self, source, background_kwargs):
        """
        Overriding the parent method, because there will be labels.
        """
        self.activate_search(source, background_kwargs, altered_param=('size', 6, 1, 3))
        self.activate_search(source, background_kwargs, altered_param=('fill_alpha', 0.6, 0.0, 0.3))

    def plot(self, label, **kwargs):
        """
        Plot the margins about a single label.
        """

        # prepare plot settings
        axes = ("x", "y")
        legend = f"{label}"
        template_kwargs = {"line_alpha": 0.6, "fill_alpha": 0.0, "size": 5}
        template_kwargs.update(kwargs)

        # create agreement/increment/decrement subsets
        col_a_pos = self.df_train[self.label_col_a] == label
        col_a_neg = self.df_train[self.label_col_a] != label
        col_b_pos = self.df_train[self.label_col_b] == label
        col_b_neg = self.df_train[self.label_col_b] != label
        agreement_slice = self.df_train[col_a_pos][col_b_pos]
        increment_slice = self.df_train[col_a_neg][col_b_pos]
        decrement_slice = self.df_train[col_a_pos][col_b_neg]
        to_plot = [
            {'source': ColumnDataSource(agreement_slice), 'marker': self.figure.square},
            {'source': ColumnDataSource(increment_slice), 'marker': self.figure.x},
            {'source': ColumnDataSource(decrement_slice), 'marker': self.figure.cross},
        ]
            
        # plot created subsets
        for _dict in to_plot:
            _source = _dict['source']
            _marker = _dict['marker']
            _kwargs = deepcopy(template_kwargs)
            self.activate_search(_source, _kwargs)
            _marker(
                *axes,
                source=_source,
                legend_label=legend,
                **_kwargs,
            )


class BokehSnorkelExplorer(BokehCorpusExplorer):
    """
    Plot text data points along with labeling function outputs.
    """

    def __init__(self, df_train, df_dev, **kwargs):
        super().__init__(df_train, **kwargs)

        # add 'label' column to df_train
        self.df_train["label"] = "unlabeled"

        # prepare plot-ready dataframe for dev set
        for _key in ["text", "label", "x", "y"]:
            assert _key in df_dev.columns
        self.df_dev = df_dev.copy()

        # keep dev set ground truth
        self.y_dev = np.array(self.df_dev[ENCODED_LABEL_KEY].tolist())

        # initialize a list of LFs to enable toggles
        self.lfs = []

    def _activate_search_on_corpus(self, source, background_kwargs):
        """
        Overriding the parent method, because there will be a labeled dev set.
        """
        self.activate_search(source, background_kwargs, altered_param=('size', 6, 1, 3))
        self.activate_search(source, background_kwargs, altered_param=('fill_alpha', 0.6, 0.0, 0.3))

    def plot(self, lf, L_train, L_dev, include, **kwargs):
        """
        Plot a single labeling function.
        """
        # keep track of added LF
        self.lfs.append(lf)

        # prepare plot settings
        axes = ("x", "y")
        decoded_targets = [lf.label_decoder[_target] for _target in lf.targets]
        legend = f"{', '.join(decoded_targets)} | {lf.name}"
        template_kwargs = {"line_alpha": 0.6, "fill_alpha": 0.0, "size": 5}
        template_kwargs.update(kwargs)

        # create correct/incorrect/missed/hit subsets
        to_plot = []
        if "C" in include:
            to_plot.append({'source': self._source_correct(L_dev), 'marker': self.figure.square})
        if "I" in include:
            to_plot.append({'source': self._source_incorrect(L_dev), 'marker': self.figure.x})
        if "M" in include:
            to_plot.append({'source': self._source_missed(L_dev, lf.targets), 'marker': self.figure.cross})
        if "H" in include:
            to_plot.append({'source': self._source_hit(L_train), 'marker': self.figure.circle})
            
        # plot created subsets
        for _dict in to_plot:
            _source = _dict['source']
            _marker = _dict['marker']
            _kwargs = deepcopy(template_kwargs)
            self.activate_search(_source, _kwargs)
            _marker(
                *axes,
                source=_source,
                legend_label=legend,
                **_kwargs,
            )

    def _source_correct(self, L_dev):
        """
        Determine the subset correctly labeled by a labeling function.
        """
        indices = self.y_dev == L_dev
        return ColumnDataSource(self.df_dev.iloc[indices])

    def _source_incorrect(self, L_dev):
        """
        Determine the subset incorrectly labeled by a labeling function.
        """
        disagreed = self.y_dev != L_dev
        attempted = L_dev != ABSTAIN_ENCODED
        indices = np.multiply(disagreed, attempted)
        return ColumnDataSource(self.df_dev.iloc[indices])

    def _source_missed(self, L_dev, targets):
        """
        Determine the subset missed by a labeling function.
        """
        targetable = np.isin(self.y_dev, targets)
        abstained = L_dev == ABSTAIN_ENCODED
        indices = np.multiply(targetable, abstained)
        return ColumnDataSource(self.df_dev.iloc[indices])

    def _source_hit(self, L_train):
        """
        Determine the subset hit by a labeling function.
        """
        indices = L_train != ABSTAIN_ENCODED
        return ColumnDataSource(self.df_train.iloc[indices])


class BokehSliderAnimation(BokehForLabeledText):
    """
    Use a 'step' slider to view an interactive animation.
    Restricted to 2D.
    """

    def __init__(self, num_steps, **kwargs):
        from bokeh.models import Slider

        assert isinstance(num_steps, int)
        self.num_steps = num_steps
        self.slider = Slider(start=0, end=self.num_steps - 1, value=0, step=1, title="Step")
        super().__init__(**kwargs)
        
    def setup_widgets(self):
        self.widgets.append(self.slider)
        super().setup_widgets()
        
    def label_steps(self, step_labels):
        from bokeh.models import Label
        assert len(step_labels) == self.num_steps, f"Expected {self.num_steps} annotations, got {len(step_labels)}"
        slider_label = Label(
            x=20, y=20, x_units='screen', y_units='screen',
            text=step_labels[0], render_mode='css',
            border_line_color='black', border_line_alpha=1.0,
            background_fill_color='white', background_fill_alpha=1.0,
        )
        slider_callback = CustomJS(
            args={
                "label": slider_label,
                "step": self.slider,
                "step_labels": step_labels,
            },
            code="""
            const S = Math.round(step.value);
            label.text = step_labels[S];
            label.change.emit();
        """,
        )

        self.slider.js_on_change("value", slider_callback)
        self.figure.add_layout(slider_label)
    
    def plot(self, traj_arr, attribute_data, method, **kwargs):
        """
        Add a plot to the figure.
        :param traj_arr: steps-by-point-by-dim array.
        :type traj_arr: numpy.ndarray with shape (self.slider.end, num_data_point, 2)
        :param attribute_data: {column->list} mapping, e.g. 'list' orientation of a Pandas DataFrame.
        :type attribute_data: dict
        :param method: the name of the plotting method to call.
        :type method: str, e.g. 'circle', 'square'
        """
        num_steps, num_data_point, num_dim = traj_arr.shape
        assert num_steps == self.num_steps, f"Expected {self.num_steps} steps, got {num_steps}"
        
        # require text and label attributes to be present
        assert "text" in attribute_data
        assert "label" in attribute_data

        # make a copy of attributes
        data_dict = attribute_data.copy()
        for _key, _value in data_dict.items():
            assert (
                len(_value) == num_data_point
            ), f"Length mismatch: {len(_value)} vs. num_data_point"

        coords_dict = {
            "x": traj_arr[0, :, 0],
            "y": traj_arr[0, :, 1],
            "xt": traj_arr[:, :, 0].flatten(),
            "yt": traj_arr[:, :, 1].flatten(),
        }
        data_dict.update(coords_dict)

        source = ColumnDataSource(data=data_dict)
        self.activate_search(source, kwargs)
        
        scatter = getattr(self.figure, method)
        scatter("x", "y", source=source, **kwargs)

        slider_callback = CustomJS(
            args={"source": source, "step": self.slider},
            code="""
            const data = source.data;
            const S = Math.round(step.value);
            var x = data['x'];
            var y = data['y'];
            for (var i = 0; i < x.length; i++) {
                x[i] = data['xt'][S * x.length + i];
                y[i] = data['yt'][S * x.length + i];
            }
            source.change.emit();
        """,
        )

        self.slider.js_on_change("value", slider_callback)