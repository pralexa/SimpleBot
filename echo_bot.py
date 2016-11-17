
import tensorflow as tf
import bot

class Echo(bot.Bot):

    def init_for_conversation(self):
        return

    def init_and_train(self):
        raise RuntimeError("Echo bot cannot be trained!")

    def get_response(self, sentence):
        return sentence

def main(_):
    bot = Echo(mode='converse')
    bot.converse()


if __name__ == "__main__":
    tf.app.run()