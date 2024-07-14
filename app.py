from flask import Flask, request, render_template
import gpt_2_simple as gpt2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='run_scifi')
    generated_text = gpt2.generate(sess,
                                   run_name='run_scifi',
                                   prefix=prompt,
                                   return_as_list=True,
                                   length=200)[0]
    return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)

