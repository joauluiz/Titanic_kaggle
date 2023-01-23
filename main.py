import time

from user import user_model_choose, user_inputs


def main():
    acc, model = user_model_choose()

    while True:
        inputs = user_inputs()

        output_model = model.predict(inputs)

        if output_model[0] == 1:
            print("\nThe model result is: Survived\n")
            teste = "The model result is: Survived"

        else:
            print("\nThe model result is: Died\n")
            teste = "The model result is: Died"
        time.sleep(2)

        new_inputs = input("Would you like to try new inputs?\nType 0 - Yes\nType 1 - No\nType: ")

        if new_inputs == '0':
            time.sleep(0)

        elif new_inputs == '1':
            break

    print("\nEnd of code")
    return teste


if __name__ == '__main__':
    main()
