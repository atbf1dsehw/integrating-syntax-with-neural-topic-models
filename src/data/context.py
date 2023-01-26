def make_context_vector(context_type: str,
                        context_size: int,
                        data: list):
    """
    This function adds context vector to the data.
    Current two types of context are supported:
    1. symmetric: from left and right
    2. asymmetric: from left only
    
    Arguments
    ---------
    context_type: str
        Type of context vector to be added.
    context_size: int
        Size of context vector.
    data: list
        List of documents. (list of list of words)
        
    Returns
    -------
    context_vec: list
        List of context vectors. (list of list of list of words)
    context_target: list
        List of target words. (list of list of words)
    """
    if context_type == 'symmetric':
        context_vec = []
        context_target = []
        for i in data:
            context_doc = []
            target_doc = []
            for j in range(len(i)):
                if j >= context_size and j < len(i) - context_size:
                    context_doc.append(i[j - context_size:j] + i[j + 1:j + context_size + 1])
                    target_doc.append(i[j])
            context_vec.append(context_doc)
            context_target.append(target_doc)

    elif context_type == 'asymmetric':
        context_vec = []
        context_target = []
        for i in data:
            context_doc = []
            target_doc = []
            for j in range(len(i)):
                if j >= context_size and j < len(i) - context_size:
                    context_doc.append(i[j - context_size:j])
                    target_doc.append(i[j])
            context_vec.append(context_doc)
            context_target.append(target_doc)
    # now flatten the context_vec and context_target
    context_vec = [item for sublist in context_vec for item in sublist]
    context_target = [item for sublist in context_target for item in sublist]
    return context_vec, context_target


if __name__ == '__main__':
    doc = ['a', 'b', 'c', 'd', 'e', 'f']
    data = [doc, ['a', 'b', 'c', 'd']]
    context_vec, context_target = make_context_vector('symmetric', 1, data)
    print(context_vec)
    print(context_target)
