def autograd(operations: list, input_values: list) -> dict:
    """
    Returns the final scalar and gradients of the initial inputs.
    """
    values = [float(value) for value in input_values]
    records = []
    for operation in operations:
        name = operation[0]
        if name == "add":
            left, right = operation[1], operation[2]
            values.append(values[left] + values[right])
            records.append((name, left, right, len(values)-1))
    
        elif name == "mul":
            left, right = operation[1], operation[2]
            values.append(values[left] * values[right])
            records.append((name, left, right, len(values)-1))
    
        else:
            parent = operation[1]
            values.append(-values[parent])
            records.append((name, parent, None, len(values)-1))
    
    gradients = [0.0] * len(values)
    gradients[-1] = 1.0

    for name, left, right, output_index in reversed(records):
        upstream = gradients[output_index]
        
        if name == "add":
            gradients[left] += upstream
            gradients[right] += upstream
        elif name == "mul":
            gradients[left] += upstream*values[right]
            gradients[right] += upstream*values[left]
        else:
            gradients[left]-=upstream

    return {"output":round(values[-1],4),"gradients":[round(value,4) for value in gradients[:len(input_values)]]}