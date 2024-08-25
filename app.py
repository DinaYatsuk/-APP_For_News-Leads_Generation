from flask import Flask, render_template, request
from core.scripts import GPT2Model, GPT



app = Flask(__name__)
app.secret_key = "0"
app.run(debug=True, threaded=True)

MODEL_PATH = "DinaYatsuk/rugpt3medium_TASS"

my_model = GPT2Model(MODEL_PATH)
model = GPT()


@app.route('/', methods=('GET', 'POST'))
async def index():
    answer = None
    errors = None

    if request.method == 'POST':
        text = request.form.get('text')
        text = text.replace('\n', '')
        text = text.replace('\t', '')

        selected_model = request.form.get('model')  # Получаем выбранную модель из формы
        if selected_model:
            if text:
                if selected_model == 'rugpt3medium':
                    answer = my_model.generate_summary_by_text(text)
                elif selected_model == 'gpt3.5':
                    answer = await model.get_gpt3(text)
                    print(answer) # Await the coroutine object and get the text
                elif selected_model == 'gpt4': # You may need to adjust how you pass the input to gpt4
                    answer = await model.get_gpt4(text)
            else:
                errors = ['Вы не задали текст!']

    return render_template("index.html", answer=answer, errors=errors)
