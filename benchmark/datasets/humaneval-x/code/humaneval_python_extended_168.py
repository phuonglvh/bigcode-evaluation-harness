import tkinter as tk

def create_form():
    """
    Create a simple login form with fields for username and password, and a submit button.
    When the submit button is pressed, 'Submitted!' is printed to the console.
    """
    window = tk.Tk()
    window.title('Login Form')

    username_label = tk.Label(window, text='Username:')
    username_label.grid(row=0, column=0)
    username_entry = tk.Entry(window)
    username_entry.grid(row=0, column=1)

    password_label = tk.Label(window, text='Password:')
    password_label.grid(row=1, column=0)
    password_entry = tk.Entry(window, show='*')
    password_entry.grid(row=1, column=1)

    submit_button = tk.Button(window, text='Submit', command=lambda: print('Submitted!'))
    submit_button.grid(row=2, columnspan=2)

    # auto close after xs
    window.after(10000, window.destroy)
    window.mainloop()


# Test if the form can be created
def test_create_form():
    try:
        create_form()
        assert True  # If no exception is thrown, the form is created successfully
    except Exception as e:
        print(f'Error: {e}')
        assert False

test_create_form()
