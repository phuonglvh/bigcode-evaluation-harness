def write_hello_world(filename: str) -> None:
    """ Write 'Hello world' to the specified file. 
    >>> write_hello_world('hello_world.txt')
    """
    with open(filename, 'w') as file:
        file.write('Hello world')


def check(write_hello_world):
    write_hello_world('hello_world.txt')
    with open('hello_world.txt', 'r') as file:
        content = file.read()
    assert content == 'Hello world'


check(write_hello_world)
