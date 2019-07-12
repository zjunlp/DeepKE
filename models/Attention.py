import torch
import torch.nn as nn


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(200)
         >>> query = torch.randn(4, 1, 200)
         >>> context = torch.randn(4, 10, 200)
         >>> inputs_len = [10, 8, 5, 5]
         >>> output, weights = attention(query, context, inputs_len)
         >>> output.size()
         torch.Size([4, 1, 200])
         >>> weights.size()
         torch.Size([4, 1, 10])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, inputs_len=None):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        query_len = context.size(1)
        if self.attention_type == "general":
            context = self.linear_in(context)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query,
                                     context.transpose(1, 2).contiguous())

        # Include mask on PADDING_INDEX
        if inputs_len:
            max_len = query_len
            mask_len = []
            for i in inputs_len:
                row = [0] * i + [1] * (max_len - i)
                mask_len.append(row)
            mask_len = torch.ByteTensor(mask_len)
            mask_len = mask_len.unsqueeze(1)
            attention_scores.masked_fill_(mask_len, float('-inf'))

        # Compute weights across every context sequence
        attention_weights = torch.softmax(attention_scores, -1)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=-1)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined)
        output = self.tanh(output)

        return output, attention_weights


if __name__ == '__main__':
    import random
    attention = Attention(200)
    query = torch.randn(4, 1, 200)  # B * L * D    hn
    context = torch.randn(4, 5, 200)  # B * Q * D    en_out
    inputs_len = [random.randint(1, 5) for _ in range(4)]
    print(inputs_len)
    output, weights = attention(query, context, inputs_len)
    print(output.shape)
    # torch.Size([64, 1, 200])
    print(weights.shape)
    # torch.Size([64, 1, 10])
    print(weights.data)
