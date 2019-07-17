import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(torch.nn.Module):
    """ A combination of multiple convolution layers and max pooling layers.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    **Thank you** to AI2 for their initial implementation of :class:`CNN`. Here is
    their `License
    <https://github.com/allenai/allennlp/blob/master/LICENSE>`__.

    Args:
        embedding_dim (int): This is the input dimension to the encoder.  We need this because we
          can't do shape inference in pytorch, and we need to know what size filters to construct
          in the CNN.
        num_filters (int): This is the output dim for each convolutional layer, which is the number
          of "filters" learned by that layer.
        ngram_filter_sizes (:class:`tuple` of :class:`int`, optional): This specifies both the
          number of convolutional layers we will create and their sizes. The default of
          ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding ngrams of
          size 2 to 5 with some number of filters.
        conv_layer_activation (torch.nn.Module, optional): Activation to use after the convolution
          layers.
        output_dim (int or None, optional) : After doing convolutions and pooling, we'll project the
          collected features into a vector of this size.  If this value is ``None``, we will just
          return the result of the max pooling, giving an output of shape
          ``len(ngram_filter_sizes) * num_filters``.
    """

    def __init__(self,
                 embedding_dim,
                 num_filters,
                 ngram_filter_sizes=(3, 4, 5),
                 output_dim=None):
        super(CNN, self).__init__()
        self._output_dim = output_dim
        self._convolution_layers = [
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=ngram_size)
            for ngram_size in ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = num_filters * len(ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim,
                                              self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (:class:`torch.FloatTensor` [batch_size, num_tokens, input_dim]): Sequence
                matrix to encode.
            mask (:class:`torch.FloatTensor`): Broadcastable matrix to `tokens` used as a mask.
        Returns:
            (:class:`torch.FloatTensor` [batch_size, output_dim]): Encoding of sequence.
        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(
                F.relu(convolution_layer(tokens)).max(dim=-1)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape
        # `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(
            filter_outputs,
            dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result


if __name__ == '__main__':
    torch.manual_seed(1)
    model = CNN(embedding_dim=200, num_filters=100)
    input = torch.randn(32, 20, 200)  # batch_size x seq_len x embedding_size
    out = model(input)
    print(out.shape)  # batch_size * (3 * num_filters)
    # print(out.data)
