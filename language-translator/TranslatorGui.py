
"""
This class is a GUI for language translation.
"""


from tkinter import *
import pickle
import numpy as np
from keras.models import load_model
from keras.utils.data_utils import pad_sequences


class translatorGui:

    def __init__(self) -> None:
        """
         initialize tkinter window and load the file
        """
        self.window = Tk()
        self.main_window()
        self.datafile()

    def datafile(self) -> None:
        """
          get all datas from datafile and load the model.
        :return:
        """
        datafile = pickle.load(open("training_data.pkl", "rb"))
        self.y_tk = datafile['y_ty']
        self.x_tk = datafile['x_tk']
        self.x = datafile['x']
        self.model = load_model("model.h5")

    def run(self) -> None:
        """
        run window

        :return:void
        """
        self.window.mainloop()

    def main_window(self) -> None:
        """
        add title to window and configure it
        :return:
        """
        self.window.title("Language Translator - RNN")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=520, height=520, bg="#000")

        head_label = Label(self.window, bg="#000", fg="#FFF", text="Welcome to Language Translator",
                           font="Melvetica 13 bold",
                           pady=10)
        head_label.place(relwidth=1)
        line = Label(self.window, width=450, bg="#000")
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # TODO: create text widget where input and output will be displayed
        self.text_widget = Text(self.window, width=20, height=2, bg="#fff", fg="#000", font="Melvetica 14", padx=5,
                                pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # TODO: create scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # TODO: create bottom label where text widget will placed
        bottom_label = Label(self.window, bg="#ABB2B9", height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        # TODO: this is for user to put english text
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg="#FFF", font="Melvetica 14")
        self.msg_entry.place(relwidth=0.788, relheight=0.06, rely=0.008, relx=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.on_enter)
        # TODO: send button which will call on_enter function to send the text
        send_button = Button(bottom_label, text="Send", font="Melvetica 13 bold", width=8, bg="#fff",
                             command=lambda: self.on_enter(None))
        send_button.place(relx=0.80, rely=0.008, relheight=0.06, relwidth=0.20)

    def decode_sequence(self, input_seq: str) -> str:
        """
        decode
        :param input_seq: string before decoding
        :return: string after decoding
        """
        try:
            y_id_to_word = {value: key for key, value in self.y_tk.word_index.items()}
            y_id_to_word[0] = ''
            sentence = input_seq
            sentence = [self.x_tk.word_index[word] for word in sentence.split()]
            sentence = pad_sequences([sentence], maxlen=self.x.shape[-1], padding='post')
            sentences = np.array([sentence[0], self.x[0]])
            predictions = self.model.predict(sentences, len(sentences))
            return ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])

        except:
            return "Error! can't translate"

    def on_enter(self, event) -> None:
        """
        get user query and bot response
        :param event:
        :return: void
        """
        msg = self.msg_entry.get()
        self.my_msg(msg, "English")
        self.deocded_output(msg, "Decoded")

    def deocded_output(self, msg: str, sender: str) -> None:
        """
        :param msg:
        :param sender:
        :return:
        """
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, str(sender) + " : " + self.decode_sequence(msg.lower())
                                + "\n\n")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)

    def my_msg(self, msg: str, sender: str) -> None:
        """
        :param msg:
        :param sender:
        :return:
        """
        if not msg:
            return
        self.msg_entry.delete(0, END)
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, str(sender) + " : " + str(msg) + "\n")
        self.text_widget.configure(state=DISABLED)


# TODO: run the file
if __name__ == "__main__":
    gui = translatorGui()
    gui.run()
