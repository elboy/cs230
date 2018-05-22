"""Define the model."""

import tensorflow as tf


def build_model_rnn(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    sentence = inputs['sentence']

    if params.model_version == 'lstm':
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size])
        sentence = tf.nn.embedding_lookup(embeddings, sentence)

        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        output, _  = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)

        # Compute logits from the output of the LSTM
        logits = tf.layers.dense(output, params.number_of_tags)

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits


def build_model_linear(mode, inputs, params):
    """Compute linear regression of the model

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    features = inputs['features']
    
    preds = tf.contrib.layers.fully_connected(features, 1, activation_fn = None)
    #preds = tf.layers.dense(features, 1)
    return preds

def build_model_fcn(mode, inputs, params):
    """Compute linear regression of the model

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    features = inputs['features']
    
    A1 = tf.contrib.layers.fully_connected(features, 200, activation_fn = tf.nn.relu)
    A2 = tf.contrib.layers.fully_connected(features, 200, activation_fn = tf.nn.relu)
    preds = tf.contrib.layers.fully_connected(features, 1, activation_fn = None)
    return preds


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    features = inputs['features']
    homeruns = inputs['homeruns']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        #predictions = build_model_linear(mode, inputs, params)
        predictions = build_model_linear(mode, inputs, params)

    # Define loss
    #loss = tf.reduce_sum(tf.pow(pred-homeruns, 2))/(2*n_samples)
    loss = tf.losses.mean_squared_error(homeruns, predictions)
    """
    OMIT
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.sequence_mask(sentence_lengths)
    losses = tf.boolean_mask(losses, mask)
    loss = tf.reduce_mean(losses)
    """
    correlation = tf.contrib.metrics.streaming_pearson_correlation(predictions, homeruns)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'correlation': tf.contrib.metrics.streaming_pearson_correlation(labels=homeruns, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    #tf.summary.scalar('correlation', correlation)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['correlation'] = correlation
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
