import speech_recognition as sr
import os
import csv
import uuid
import pyttsx3
from flask import Flask, render_template, request, url_for, redirect, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func, desc
from datetime import date
import datetime
import time
import ast

from gingerit.gingerit import GingerIt

app = Flask(__name__)

app.config[
    "CSV-UPLOADS"] = "C:\\Users\\Tjandra Putra\\Documents\\Final Year Project - P2\\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads"

app.config['SECRET_KEY'] = '878BC1623EE36E4E8ED239C62B672'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot_user_data.db'

db = SQLAlchemy(app)  # SQl instance

# todo: ============= DATABASE STRUCTURE =============

class Messages(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message_unknown = db.Column(db.String(200))
    isArchived = db.Column(db.Integer, default=0)
    tag = db.Column(db.String(200), default="")
    admin_response = db.Column(db.String(200), default="")

class Ranking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Integer)
    username = db.Column(db.String(20), default="Anonymous")

class Files(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fileName = db.Column(db.String(100))
    date = db.Column(db.String(20))

class Quizes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fileName = db.Column(db.String(100))
    date = db.Column(db.String(20))

class savedQuizes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(100))
    answer = db.Column(db.String(200))
    isSelected = db.Column(db.Integer, default=0)
    options = db.Column(db.String(500))

class training_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(200), default="")
    pattern = db.Column(db.String(200))
    response = db.Column(db.String(200), default="")
    frequency = db.Column(db.Integer, default=0) # To see how often users ask this question

# todo: ============= DATABASE STRUCTURE ENDS =============

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

bot_name = "Bot"

with open("training_data.json") as file:
    data = json.load(file)

words = []  # List of words
labels = []  # List of tags
docs_x = []  # List of different patterns
docs_y = []  # List of tags corresponding to docs_x

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))  # remove duplicate words

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#todo:  ============== Training & Saving model ==============

model_name = "model.tflearn"

def train_model():
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save(model_name)

# Back up reset in case I manually delete from the json file.
# train_model()

# todo: ============== Loading model ==============

def load_model():
    model.load(model_name)

load_model()

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

# todo: ===================== PREPROCESSING ENDS ========================

@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    all_data = Ranking.query.order_by(desc(Ranking.score))

    quiz_data = savedQuizes.query.filter_by(isSelected=1)

    total_questions = str(quiz_data.count())

    return render_template("index.html", all_data=all_data, total_questions=total_questions)

# todo: Clearing leaderboard table
@app.route("/reset_score")
def reset_score():
    all_data = Ranking.query.all()

    for row in all_data:
        db.session.delete(row)
        db.session.commit()

    flash("Leaderboard has been reset successfully.", "success")

    return redirect(url_for('admin_quiz'))


@app.route("/admin_quiz", methods=['GET', 'POST'])
def admin_quiz():
    import ast

    page = request.args.get('page_file', 1, type=int)
    file_data = Quizes.query.filter_by(id=Quizes.id).paginate(page=page, per_page=5)

    page = request.args.get('page', 1, type=int)
    quiz_data = (savedQuizes.query.filter_by(id=savedQuizes.id)).paginate(page=page, per_page=5)

    database = savedQuizes.query.filter_by(isSelected=1)
    total_selected = database.count()

    # Displaying all questions uploaded
    return render_template("admin_quiz.html", file_data=file_data, quiz_data=quiz_data, total_selected=total_selected, ast=ast)


# todo: Deleting for table in admin_quiz.html - Upload Files
@app.route('/delete_model_quiz/<id>/', methods=['GET', 'POST'])
def delete_quiz(id):
    file_data = savedQuizes.query.get(id)
    db.session.delete(file_data)
    db.session.commit()

    flash("Row {} has been deleted successfully!".format(file_data.id), "success")

    return redirect(url_for('admin_quiz'))

# todo: Uploading quiz .csv for admin_quiz.html
@app.route("/upload_quiz", methods=['GET', 'POST'])
def upload_quiz():
    if request.method == 'POST':

        fileName_list = []

        database = Quizes.query.all()

        for row in database:
            fileName_list.append(row.fileName)

        if request.files['file-csv'].filename in fileName_list:
            flash("This file name '{}' has been taken, please select a new file name.".format(request.files['file-csv'].filename), "danger")

        # Checks if upload file is empty
        elif request.files['file-csv'].filename == '':
            flash("Please upload a file!", "danger")

        elif not (request.files['file-csv'].filename).endswith('.csv'):
            flash("Please upload .CSV file only!", "danger")

        # Get file from the user
        elif request.files:
            csv_input = request.files['file-csv']  # Object
            csv_input.save(os.path.join("C:\\Users\\Tjandra Putra\\Documents\\Final Year Project - P2\\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads_quiz", csv_input.filename))

            # dd/mm/YY
            today = date.today()
            d1 = today.strftime("%d/%m/%Y")

            current_time = datetime.datetime.now()

            format_time = str(d1) + ", " + str(current_time.strftime("%I")) + ":" + str(
                current_time.strftime("%M")) + " " + str(current_time.strftime("%p"))

            file_saved = Quizes(fileName=csv_input.filename, date=format_time)

            db.session.add(file_saved)
            db.session.commit()

            # Saving to the database display all quizes
            directory = "C:\\Users\\Tjandra Putra\\Documents\\Final Year Project - P2\\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads_quiz\\"
            complete_path = directory + csv_input.filename
            with open(complete_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)  # Skips header

                for row in csvreader:
                    list_option = []

                    list_option.append(row[2])
                    list_option.append(row[3])
                    list_option.append(row[4])
                    database = savedQuizes(question=row[0], answer=row[1], options=str(list_option))
                    db.session.add(database)
                    db.session.commit()

                flash("File has been uploaded", "success")

            return redirect(url_for('admin_quiz'))

    return redirect(url_for('admin_quiz'))


# todo: Deleting for table in admin_quiz.html - Upload Files
@app.route('/delete_model_quiz_files/<id>/', methods=['GET', 'POST'])
def delete_files_quiz(id):
    file_data = Quizes.query.get(id)
    db.session.delete(file_data)
    db.session.commit()

    # Remove from folder
    file_path = "C:\\Users\\Tjandra Putra\\Documents\\Final Year Project - P2\\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads_quiz\\"
    os.remove(file_path + file_data.fileName)

    flash("Row {} has been deleted successfully!".format(file_data.id), "success")

    return redirect(url_for('admin_quiz'))


@app.route('/download_quiz/<id>')
def downloadFile_quiz(id):
    download_id = Quizes.query.get(id)

    file_name = download_id.fileName

    file_path = os.path.join("C:\\Users\\Tjandra Putra\\Documents\\Final Year Project - P2\\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads_quiz", file_name)

    return send_file(file_path, as_attachment=True)


@app.route("/display_quiz_no/<id>", methods=['GET', 'POST'])
def display_quiz_no(id):

    all_data = savedQuizes.query.get(id)

    all_data.isSelected = 0
    db.session.commit()

    return redirect(url_for('admin_quiz'))


@app.route("/display_quiz_yes/<id>", methods=['GET', 'POST'])
def display_quiz_yes(id):

    all_data = savedQuizes.query.get(id)

    all_data.isSelected = 1
    db.session.commit()

    return redirect(url_for('admin_quiz'))


@app.route("/add_quiz", methods=['GET', 'POST'])
def add_quiz():
    if request.method == 'POST':
        options_list = []

        user_question = request.form['question']
        user_answer = request.form['answer']
        user_option_1 = request.form['option_1']
        user_option_2 = request.form['option_2']
        user_option_3 = request.form['option_3']

        options_list.append(user_option_1)
        options_list.append(user_option_2)
        options_list.append(user_option_3)

        obj_quiz = savedQuizes(question=user_question, answer=user_answer, options=str(options_list))
        db.session.add(obj_quiz)
        db.session.commit()

        flash("Quiz has been added successfully!", "success")

    return redirect(url_for('admin_quiz'))


# todo: Quiz page all questions
@app.route("/quiz", methods=['GET', 'POST'])
def quiz():
    database = savedQuizes.query.filter_by(isSelected=1)
    dictionary_data = {}
    answer_key = []
    total_score = 0
    global_username = ''

    # A list of questions and option are be stored into dictionary for unique
    for row in database:
        answer_key.append(row.answer)
        options_list = ast.literal_eval(row.options)

        # append to dictionary
        dictionary_data[row.question] = options_list
        print(dictionary_data)

    if request.method == "POST":
        username = request.form['username']
        global_username = username

        for key, value in dictionary_data.items():
               answer_user = request.form.get(key)
               if answer_user in answer_key:
                   total_score += 1

        user = Ranking(username=global_username, score=total_score)
        db.session.add(user)
        db.session.commit()

        if total_score < 3:
            flash("You got a score of  " + str(total_score) + "/" + str(database.count()) + ". It's okay practice makes perfect :)", "info")

        elif total_score >= 3:
            flash("You got a score of  " + str(total_score) + "/" + str(database.count()) + ". Well done! Keep up the great work !", "info")

    return render_template("quiz.html", database=database, dictionary_data=dictionary_data)


# todo: Append existing row for admin_chatbot.html
@app.route("/append_data/", methods=['GET', 'POST'])
def append_data():
    messages_table = Messages.query.all()
    ERROR = False
    for messages_row in messages_table:
        if messages_row.tag == "":
            ERROR = True
            break

    if ERROR:
        flash("Please ensure that the tags are not empty!", "danger")
    else:
        if request.method == 'POST':
            random_id = uuid.uuid1()
        csv_file_name = 'GENERATED_' + str(random_id) + '.csv'

        messages_data = Messages.query.filter_by(isArchived=0)

        file_path = "C:\\Users\\Tjandra Putra\Documents\\Final Year Project - P2\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads"

        complete_path = os.path.join(file_path, csv_file_name)

        with open(complete_path, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["tag", "patterns", "responses"])

            for x in messages_data:
                writer.writerow([x.tag, x.message_unknown, x.admin_response])

        # dd/mm/YY
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")

        current_time = datetime.datetime.now()

        format_time = str(d1) + ", " + str(current_time.strftime("%I")) + ":" + str(
            current_time.strftime("%M")) + " " + str(current_time.strftime("%p"))

        file_saved = Files(fileName=csv_file_name, date=format_time)
        db.session.add(file_saved)
        db.session.commit()

        # Archive
        messages_data = Messages.query.all()
        for x in messages_data:
            x.isArchived = 1
            db.session.commit()

        # =========== MAIN PROCESS ===========
        with open(complete_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            next(csvreader)  # Skips header

            json_tag_list = []

            # Storing json tags in a list
            with open('training_data.json', 'r') as f:
                data = json.load(f)

                for x in data['intents']:
                    json_tag_list.append(x['tag'])

            # Checks if csv tag exists in the json list
            for row in csvreader:
                if row[0] in json_tag_list:  # Checks if tag exists, if yes, it needs to find out its index

                    with open('training_data.json', 'r') as f:
                        data = json.load(f)
                        for x in data['intents']:

                            if x['tag'] == row[0]:
                                x['patterns'].append(row[1])

                                # Also add to database table "training_data"
                                training_data_db = training_data.query.all()

                                for db_row in training_data_db:
                                    if db_row.tag == row[0]:
                                        # Convert string list to type list
                                        list_pattern = ast.literal_eval(db_row.pattern)
                                        # Appending to list
                                        list_pattern.append(row[1])
                                        # Updating database
                                        db_row.pattern = str(list_pattern)
                                        db.session.commit()

                                print("Added to existing tag: {}".format(row[0]))

                            else:
                                continue  # Dont do anything

                    os.remove('training_data.json')
                    with open('training_data.json', 'w') as f:
                        json.dump(data, f, indent=4)

                else:  # Appends new dictionary category tag
                    new_json_category = {"tag": row[0], "patterns": [row[1]], "responses": [row[2]]}

                    data['intents'].append(new_json_category)

                    print("Added New Category: {}".format(row[0]))

                    os.remove('training_data.json')
                    with open('training_data.json', 'w') as f:
                        json.dump(data, f, indent=4)

                    # Adding to datbase also
                    pattern_list = []

                    pattern_list.append(row[1])

                    db_obj = training_data(tag=row[0], pattern=str(pattern_list), response=row[2])
                    db.session.add(db_obj)
                    db.session.commit()

        flash("Rows have been appended to the training data successfully.", "success")

    return redirect(url_for('table'))


# todo: Down option for admin_chatbot.html
@app.route('/download/<id>')
def downloadFile(id):
    download_id = Files.query.get(id)

    file_name = download_id.fileName

    file_path = os.path.join(app.config["CSV-UPLOADS"], file_name)

    return send_file(file_path, as_attachment=True)


# todo: Upload file for admin_chatbot.html
@app.route("/upload-csv", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        fileName_list = []

        database = Files.query.all()

        for row in database:
            fileName_list.append(row.fileName)

        if request.files['file-csv'].filename in fileName_list:
            flash("The file name '{}' has been taken, please select a new file name.".format(request.files['file-csv'].filename), "danger")

        # Checks if upload file is empty
        elif request.files['file-csv'].filename == '':
            flash("Please upload a file!", "danger")

        elif not (request.files['file-csv'].filename).endswith('.csv'):
            flash("Please upload .CSV file only!", "danger")

        # Get file from the user
        elif request.files:
            csv_input = request.files['file-csv']  # Object

            csv_input.save(os.path.join(app.config["CSV-UPLOADS"], csv_input.filename))

            print(csv_input.filename)

            # dd/mm/YY
            today = date.today()
            d1 = today.strftime("%d/%m/%Y")

            current_time = datetime.datetime.now()

            format_time = str(d1) + ", " + str(current_time.strftime("%I")) + ":" + str(
                current_time.strftime("%M")) + " " + str(current_time.strftime("%p"))

            file_saved = Files(fileName=csv_input.filename, date=format_time)

            db.session.add(file_saved)
            db.session.commit()

            file_path = "C:\\Users\\Tjandra Putra\\Documents\\Final Year Project - P2\\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads"

            complete_path = os.path.join(file_path, csv_input.filename)

            # =========== MAIN PROCESS ===========
            with open(complete_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)

                next(csvreader)  # Skips header

                json_tag_list = []

                # Storing json tags in a list
                with open('training_data.json', 'r') as f:
                    data = json.load(f)

                    for x in data['intents']:
                        json_tag_list.append(x['tag'])

                # Checks if csv tag exists in the json list
                for row in csvreader:
                    if row[0] in json_tag_list:  # Checks if tag exists, if yes, it needs to find out its index

                        with open('training_data.json', 'r') as f:
                            data = json.load(f)
                            for x in data['intents']:

                                if x['tag'] == row[0]:
                                    x['patterns'].append(row[1])
                                    print("Added to existing tag: {}".format(row[0]))

                                    # Also add to database table "training_data"
                                    training_data_db = training_data.query.all()

                                    for db_row in training_data_db:
                                        if db_row.tag == row[0]:
                                            # Convert string list to type list
                                            list_pattern = ast.literal_eval(db_row.pattern)
                                            # Appending to list
                                            list_pattern.append(row[1])
                                            # Updating database
                                            db_row.pattern = str(list_pattern)
                                            db.session.commit()

                                else:
                                    continue  # Dont do anything

                        os.remove('training_data.json')
                        with open('training_data.json', 'w') as f:
                            json.dump(data, f, indent=4)

                    else:  # Appends new dictionary category tag
                        new_json_category = {"tag": row[0], "patterns": [row[1]], "responses": [row[2]]}

                        data['intents'].append(new_json_category)

                        print("Added New Category: {}".format(row[0]))

                        os.remove('training_data.json')
                        with open('training_data.json', 'w') as f:
                            json.dump(data, f, indent=4)

                        # Adding to datbase also
                        pattern_list = []

                        pattern_list.append(row[1])

                        db_obj = training_data(tag=row[0], pattern=str(pattern_list), response=row[2])
                        db.session.add(db_obj)
                        db.session.commit()

            flash("'{}' has been uploaded successfully!".format(csv_input.filename), "success")
            return redirect(url_for('table'))

    return redirect(url_for('table'))


# todo: Deleting for admin_chatbot.html table - Upload Files
@app.route('/delete_model_files/<id>/', methods=['GET', 'POST'])
def delete_files_messages(id):
    file_data = Files.query.get(id)
    db.session.delete(file_data)
    db.session.commit()

    # Remove from folder
    file_path = "C:\\Users\\Tjandra Putra\\Documents\\Final Year Project - P2\\FYPJ_ChatBot_2.0\\FYPJ_ChatBot_2.0\\static\\uploads\\"
    os.remove(file_path + file_data.fileName)

    flash("Row {} has been deleted successfully!".format(file_data.id), "success")

    return redirect(url_for('table'))


# todo: admin_chatbot.html
@app.route("/table", methods=['GET', 'POST'])
def table():
    page = request.args.get('page', 1, type=int)
    all_data = (Messages.query.filter_by(isArchived=0)).paginate(page=page, per_page=5)

    page = request.args.get('page_archived', 1, type=int)
    archived_data = (Messages.query.filter_by(isArchived=1)).paginate(page=page, per_page=5)

    # archived_data = Messages.query.filter_by(isArchived=1)

    page = request.args.get('page_file', 1, type=int)
    file_data = (Files.query.filter_by(id=Files.id)).paginate(page=page, per_page=5)

    page = request.args.get('page_training_data', 1, type=int)
    database_training_data = (training_data.query.order_by(desc(training_data.frequency))).paginate(page=page, per_page=5)
    # Ranking.query.order_by(desc(Ranking.score))

    # Display existing tag from json file
    with open("training_data.json") as f:
        data = json.load(f)

        tag_list = []

        for dictionary in data['intents']:
            tag = dictionary['tag']
            tag_list.append(tag)

    return render_template("admin_chatbot.html", all_data=all_data, archived_data=archived_data, tag_list=tag_list,
                           file_data=file_data, database_training_data=database_training_data)


# todo: Auto-correct for table
@app.route('/checker_model_messages/<id>', methods=['GET', 'POST'])
def checker_model_messages(id):
    all_data = Messages.query.get(id)

    text_message_unknown = all_data.message_unknown

    parser = GingerIt()

    results_dictionary = parser.parse(text_message_unknown)

    all_data.message_unknown = results_dictionary['result']

    db.session.commit()
    flash("Row {} has been corrected successfully!".format(all_data.id), "success")

    return redirect(url_for('table'))


# todo: Deleting for table
@app.route('/delete_model_messages/<id>/', methods=['GET', 'POST'])
def delete_model_messages(id):
    all_data = Messages.query.get(id)
    db.session.delete(all_data)
    db.session.commit()
    flash("Row {} has been deleted successfully!".format(all_data.id), "success")

    return redirect(url_for('table'))


# todo: Retrain for table
@app.route('/retrain/', methods=['GET', 'POST'])
def retrain():
    # Retrain additional confirmation
    if request.method == "POST":
        confirmation_value = request.form['retrain']
        if confirmation_value == "TRAIN/model":
            model_name = 'model.tflearn'
            existing_models_list = [model_name + '.data-00000-of-00001', model_name + '.index', model_name + '.meta']
            for item in existing_models_list:
                os.remove(item)

            with open("training_data.json") as file:
                data = json.load(file)

            words = []
            labels = []
            docs_x = []
            docs_y = []

            for intent in data["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent["tag"])

                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

            words = [stemmer.stem(w.lower()) for w in words if w != "?"]
            words = sorted(list(set(words)))

            labels = sorted(labels)

            training = []
            output = []

            out_empty = [0 for _ in range(len(labels))]

            for x, doc in enumerate(docs_x):
                bag = []

                wrds = [stemmer.stem(w.lower()) for w in doc]

                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)

            training = numpy.array(training)
            output = numpy.array(output)

            tensorflow.reset_default_graph()

            net = tflearn.input_data(shape=[None, len(training[0])])
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
            net = tflearn.regression(net)

            model = tflearn.DNN(net)

            # ============== Saving model ==============
            model_name = "model.tflearn"

            def train_model():
                model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
                model.save(model_name)

            train_model()

            # ============== Loading model ==============
            def load_model():
                model.load(model_name)

            load_model()

            model.load(model_name)

            flash("Model has been retrained successfully!", "success")

        else:
            flash("Please ensure that you type 'TRAIN/model' correctly",
                  "danger")

    return redirect(url_for('table'))


# todo: Update for table
@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        my_data = Messages.query.get(request.form['rowid'])

        my_data.tag = request.form['tag']
        my_data.admin_response = request.form['admin_response']
        my_data.message_unknown = request.form['unknown_user_responses']

        db.session.commit()
        print(request.form['tag'])
        print(request.form['admin_response'])
        flash("Row {} has been updated successfully!".format(my_data.id), "success")

        return redirect(url_for('table'))


# todo: Update for admin_quiz.html
@app.route('/update_quiz', methods=['GET', 'POST'])
def update_quiz():
    if request.method == 'POST':

        option_lists = []

        my_data = savedQuizes.query.get(request.form['rowid'])

        my_data.question = request.form['question']
        my_data.answer = request.form['answer']

        option_1 = request.form['option_1']
        option_2 = request.form['option_2']
        option_3 = request.form['option_3']

        option_lists.append(option_1)
        option_lists.append(option_2)
        option_lists.append(option_3)

        my_data.options = str(option_lists)

        db.session.commit()

        flash("Row {} has been updated successfully!".format(my_data.id), "success")
        return redirect(url_for('admin_quiz'))

@app.route("/admin_chatbot/<id>/",  methods=['GET', 'POST'])
def add_to_quiz(id):
    faq_data = training_data.query.get(id)

    # Converts to type list
    faq_pattern_list = ast.literal_eval(faq_data.pattern)

    add_to_quiz = savedQuizes(question=str(random.choice(faq_pattern_list)), options=str(list("")), answer=random.choice(ast.literal_eval(faq_data.response)))
    db.session.add(add_to_quiz)
    db.session.commit()

    # flash("You chose : {}, index : {}".format(str(faq_data.pattern), faq_data), "success")

    flash("Row {} : Quiz has been created successfully".format(faq_data.id), "success")

    return redirect(url_for('table'))
    # return render_template("admin_chatbot.html")



@app.route("/get")
def get_bot_response():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 10.0)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 25)

    # ==================================

    userText = request.args.get('msg')
    print(userText)

    results = model.predict([bag_of_words(userText, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    print(results)
    # Returns check for empty input
    if userText.lower() == "":  # checks for empty input
        error = "Please type something"
        print("Please type something")

        time.sleep(2)

        return str(error)

    # Returns check for quiz
    elif userText.lower() == "quiz":
        return redirect(url_for("quiz"))

    # Returns Prediction
    elif results[results_index] > 0.75:  # 80% confidence
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

                update_this = training_data.query.filter_by(tag=tag).first()
                update_this.frequency += 1
                db.session.commit()

        print(random.choice(responses))

        time.sleep(2)

        return str(random.choice(responses))  # Prediction

    else:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['patterns']

        new_responses = str(random.choice(responses))

        # adding to database if unknown messages
        message_obj = Messages(message_unknown=userText)
        db.session.add(message_obj)
        db.session.commit()

        error = "I'm not sure I understand. Do you mean: '{}'".format(new_responses)
        print("I don't understand, try again")

        time.sleep(2)

        return str(error)


def bot_speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 10.0)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 25)

    engine.say(text)
    engine.runAndWait()

@app.route("/speech_to_text")
def speech_to_text():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 10.0)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 25)

    global_text = ''

    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something...")
        # flash("Please say something", 'warning')

        # audio = r.listen(source)
        # text_voice = r.recognize_google(audio, language='en-IN')

        try:
            audio = r.listen(source)
            text_voice = r.recognize_google(audio, language='en-IN')
            global_text = text_voice
            flash("You have said: {}".format(text_voice), "info")


            results = model.predict([bag_of_words(text_voice, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            # Returns Prediction
            if results[results_index] > 0.75:  # 80% confidence
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                        update_this = training_data.query.filter_by(tag=tag).first()
                        update_this.frequency += 1
                        db.session.commit()

                flash("LMS Bot: " + str(random.choice(responses)), "success")
                engine.say(random.choice(responses))
                engine.runAndWait()

            else:
                message_obj = Messages(message_unknown=global_text)
                db.session.add(message_obj)
                db.session.commit()

                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['patterns']

                new_responses = str(random.choice(responses))

                flash("LMS Bot: I'm not sure I understand. Do you mean: '{}'".format(new_responses), "warning")
                engine.say("I'm not sure I understand. Do you mean: '{}'".format(new_responses))
                engine.runAndWait()

        except Exception as e:
            # flash("Error: " + str(e), "danger")

            flash("Please say something", "danger")
            print("Error: " + str(e))

    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)
