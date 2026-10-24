
from flask import Flask, render_template, request

from detector import detect_paraphrase

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {"sentence1": "", "sentence2": "", "result": None,
           "percentage": 0.0, "error": None, "p1": "", "p2": ""}

    if request.method == "POST":
        s1 = request.form.get("sentence1", "").strip()
        s2 = request.form.get("sentence2", "").strip()
        ctx["sentence1"], ctx["sentence2"] = s1, s2

        if not s1 or not s2:
            ctx["error"] = "Please enter both sentences."
        else:
            result, similarity, p1, p2 = detect_paraphrase(s1, s2)
            ctx.update(result=result, percentage=round(similarity * 100, 2),
                       p1=p1, p2=p2)

    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
