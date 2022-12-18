methods = [
    "Text to Image",
    "Image to Image",
    "Video filter",
    "Image Chain",
]

def get_option_range(selection, low, high):
    print(f"Choose {selection}")
    print(f"{low} - {high}")
    while 1:
        try:
            inpt = float(input(":"))
            if inpt >= low and inpt <= high:
                return inpt
            else:
                print("try again")
        except ValueError:
            print("try again")
    return inpt

def get_option_input(selection, options):
    print(f"Choose {selection}")
    indexes = []
    for i, m in enumerate(options):
        print(f"{i+1}: {m}")
        indexes.append(f"{i+1}")
    while 1:
        method_index = input(",".join(indexes) + ":")
        if method_index in indexes:
            return options[int(method_index)-1]
        else:
            print("try again")

def main():
    get_option_input("Method", methods)
    get_option_range("strength", 0, 1)
        


if __name__ == '__main__':
    main()