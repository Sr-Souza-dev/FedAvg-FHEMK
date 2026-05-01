def generate_message_error(params: dict, results:dict, module:str, function:str) -> str:
    """
    Generate a message for the error in the params and results.
    """
    message = f"Error in {module.upper()} on function {function.upper()}\n\n"
    message += "Params:\n"
    for key, value in params.items():
        message += f"- {key}: {value}\n"

    message += "\nResults:\n"
    for key, value in results.items():
        message += f"- {key}: {value}\n"
    message += "\n"
    message += "Please check the params and results to ensure they match.\n"
    return message