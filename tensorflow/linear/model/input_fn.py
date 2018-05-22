"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def preproc_features(string):
    string_tensor = tf.string_split([string]).values
    float_tensor = tf.string_to_number(string_tensor)
    float_tensor.set_shape([14])
    return float_tensor

def preproc_homeruns(string):
    string_tensor = tf.string_split([string]).values
    float_tensor = tf.string_to_number(string_tensor)
    float_tensor.set_shape([1])
    return float_tensor

#OMIT: def load_dataset_from_text(path_txt, vocab):
def load_dataset_from_text(path_txt, mode="features"):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load txt file, one example per line
    dataset = tf.data.TextLineDataset(path_txt)
    #print(dataset)
    # Convert line into list of tokens, splitting by white space
    if mode == "features":
        dataset = dataset.map(preproc_features)
    else:
        dataset = dataset.map(preproc_homeruns)
    #print(dataset)
    
    """
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    features = iterator.get_next()
    #print(features.shape)
    init_op = iterator.initializer 

    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(features))
    """
    """
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(3):
            print(sess.run(next_element))
    """
    # Lookup tokens to return their ids
    #OMIT: dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))

    return dataset


def input_fn(mode, features, targets, params):
    """Input function for NER

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        sentences: (tf.Dataset) yielding list of ids of words
        datasets: (tf.Dataset) yielding list of ids of tags
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1

    # Zip the sentence and the labels together
    dataset = tf.data.Dataset.zip((features, targets))

    """
    OMIT
    # Create batches and pad the sentences of different length
    padded_shapes = ((tf.TensorShape([None]),  # sentence of unknown size
                      tf.TensorShape([])),     # size(words)
                     (tf.TensorShape([None]),  # labels of unknown size
                      tf.TensorShape([])))     # size(tags)

    padding_values = ((params.id_pad_word,   # sentence padded on the right with id_pad_word
                       0),                   # size(words) -- unused
                      (params.id_pad_tag,    # labels padded on the right with id_pad_tag
                       0))                   # size(tags) -- unused
    """


    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        #.padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    (features, homeruns) = iterator.get_next()
    init_op = iterator.initializer 

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'features': features,
        'homeruns': homeruns,
        'iterator_init_op': init_op
    }

    return inputs
