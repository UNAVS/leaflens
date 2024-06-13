from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory, jsonify, make_response
from werkzeug.utils import secure_filename
import os, requests
import tensorflow as tf
import numpy as np
from PIL import Image
from joblib import load
from ultralytics import YOLO
import logging


app = Flask(__name__, instance_relative_config=True)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'assets', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/upload', methods = ['POST'])   
def upload_file():   
    if request.method == 'POST':   
        f = request.files['file'] 
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        app.logger.info("ini awal model")
        model_kelas = tf.keras.models.load_model('ensembletop5.h5')

        

        # Preprocess the uploaded image
        def preprocess_image(image_path):
            with Image.open(image_path) as img:
                img_rescaled = img.resize((256, 256))        
                img_array = np.array(img_rescaled)
                img_normalized = img_array / 255.0
                img_expand = tf.expand_dims(img_normalized,0)
                return img_expand

        final_image = preprocess_image(filepath)

        # Make prediction
        predict1 = model_kelas.predict(final_image)
        class_index = np.argmax(predict1, axis=1)[0]
        nama = tentukan_nama(class_index)
        app.logger.info("ini akhir model")

        # model = YOLO('best.pt')
        # # model = torch.hub.load('ultralytics/yolov8', 'custom', path='best.pt')
        # print("mulai predict!")
        # test = model.predict(filepath, save=True, show_labels=False, imgsz=640, iou=0.5)
        # print("selesai predict")
        
        predicted_filename ='predicted_' + filename
        predicted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], predicted_filename)

        deskripsi = determine_deskripsi(nama)
        rekomendasi = determine_rekomendasi(nama)
        prevention = determine_prevention(nama)
        print(filename, predicted_filename, deskripsi, nama, rekomendasi, prevention)
        
        # return data to template file
        return render_template("index.html", uploaded_filename = filename, predicted_filename = predicted_filename, deskripsi = deskripsi, nama = nama, rekomendasi = rekomendasi, prevention = prevention)

def tentukan_nama(class_index):
    class_names = ['Downey-Mildew', 'Early-Blight', 'Fire-Blight', 'Fusarium-Wilt', 'Late-Blight', 'Leaf-Spot', 'Powdery-Mildew', 'Rust-Leafs', 'Sehat']
    return class_names[class_index] if class_index < len(class_names) else 'Unknown'
   
def determine_rekomendasi(nama):
    if nama == 'Downey-Mildew':
        return '''Gunakan fungisida yang efektif seperti mancozeb, fosetil-Al, atau metalaksil, 
        pangkas dan buang bagian tanaman yang terinfeksi, dan pastikan tanaman memiliki sirkulasi udara yang baik.'''
    
    elif nama == 'Early-Blight':
        return '''Aplikasikan fungisida seperti chlorothalonil atau mancozeb, 
        dan buang daun yang terinfeksi untuk mencegah penyebaran.'''
    
    elif nama == 'Fire-Blight':
        return '''Potong dan buang bagian tanaman yang terinfeksi, 
        dan gunakan antibiotik seperti streptomisin pada saat musim tanam.'''
    
    elif nama == 'Fusarium-Wilt':
        return '''Gunakan fungisida berbasis benomil atau thiophanate-methyl, 
        dan buang tanaman yang terinfeksi secara menyeluruh.'''
    
    elif nama == 'Late-Blight':
        return '''Aplikasikan fungisida seperti chlorothalonil atau mancozeb secara berkala, 
        dan buang dan bakar tanaman yang terinfeksi.'''
    
    elif nama == 'Leaf-Spot':
        return '''Gunakan fungisida berbasis tembaga atau chlorothalonil, 
        dan buang daun yang terinfeksi untuk mencegah penyebaran.'''
    
    elif nama == 'Powdery-Mildew':
        return '''Aplikasikan fungisida seperti sulfur atau myclobutanil, 
        dan pangkas bagian tanaman yang terinfeksi.'''
    
    elif nama == 'Rust-Leafs':
        return '''Gunakan fungisida seperti myclobutanil atau chlorothalonil, 
        serta buang dan bakar daun yang terinfeksi.'''
    
    else:
        return '''Pastikan tanaman mendapatkan nutrisi yang cukup dengan pemupukan yang tepat, 
        tanam varietas yang sesuai dengan kondisi iklim dan tanah setempat, jaga kebersihan lahan tanam dari gulma dan sisa tanaman yang bisa menjadi sumber penyakit, 
        rotasi tanaman untuk mencegah akumulasi patogen di tanah, dan monitor tanaman secara rutin untuk mendeteksi gejala penyakit sejak dini.''' 

def determine_prevention(nama):
    if nama == 'Downey-Mildew':
        return '''Tanam varietas yang tahan terhadap Downy Mildew, 
        hindari penyiraman dari atas yang menyebabkan daun basah terlalu lama, 
        dan jaga jarak tanam yang cukup untuk meningkatkan sirkulasi udara.'''
    
    elif nama == 'Early-Blight':
        return '''Tanam varietas yang tahan terhadap Early Blight, 
        rotasi tanaman untuk mencegah akumulasi patogen di tanah, 
        dan hindari penyiraman dari atas.'''
    
    elif nama == 'Fire-Blight':
        return '''Tanam varietas yang tahan terhadap Fire Blight, 
        prune tanaman secara rutin untuk meningkatkan sirkulasi udara, 
        serta hindari pemangkasan saat kondisi basah.'''
    
    elif nama == 'Fusarium-Wilt':
        return '''Rotasi tanaman dengan tanaman yang tidak rentan terhadap Fusarium, 
        tanam varietas yang tahan terhadap Fusarium Wilt, 
        dan jaga kebersihan alat-alat pertanian.'''
    
    elif nama == 'Late-Blight':
        return '''Tanam varietas yang tahan terhadap Late Blight, 
        hindari penanaman yang terlalu rapat, 
        serta pantau cuaca dan kondisi tanaman secara rutin.'''
    
    elif nama == 'Leaf-Spot':
        return '''Tanam varietas yang tahan terhadap Leaf Spot, 
        hindari penyiraman dari atas yang membuat daun basah, 
        jaga jarak tanam untuk sirkulasi udara yang baik.'''
    
    elif nama == 'Powdery-Mildew':
        return '''Tanam varietas yang tahan terhadap Powdery Mildew, 
        jaga sirkulasi udara yang baik di sekitar tanaman, 
        dan hindari kondisi kelembaban yang tinggi.'''
    
    elif nama == 'Rust-Leafs':
        return '''Tanam varietas yang tahan terhadap Rust Leaf, 
        jaga kebersihan sekitar tanaman, 
        dan hindari penyiraman dari atas.'''
    
    else:
        return '-'
            
def determine_deskripsi(nama):
    if nama == 'Downey-Mildew':
        return '''Downy mildew disebakan oleh Plasmopara viticola kelompok oomycetes. 
        Penyakit ini bermanifestasi sebagai bercak kuning ke putih pada permukaan atas daun dewasa.'''

    elif nama == 'Early-Blight':
        return '''Early blight disebkan oleh jamur Alternaria solani. 
        Penyakit ini umumnya terjadi pada kentang dan tomat.'''
    
    elif nama == 'Fire-Blight':
        return '''Fire blight adalah penyakit bakteri patogen yang disebabkan oleh Erwinia amylovora. 
        Penyakit ini terutama mempengaruhi tanaman dari keluarga Rosaceae, seperti apel, pir dan cotoneaster.'''
    
    elif nama == 'Fusarium-Wilt':
        return '''Fusarium wilt disebabkan oleh jamur Fusarium oxysporum.
        Penyakit ini menyerang kentang, tomat dan tanaman dari keluarga Solaneceae lainnya.'''
    
    elif nama == 'Late-Blight':
        return '''Late blight disebabkan oleh jamur patogen Phytophthora infestans. 
        Penyakit ini sering terjadi selama pertumbuhan, perkembangan tanaman dan mungkin muncul setelah pembungaan.'''
    
    elif nama == 'Leaf-Spot':
        return '''Leaf spots dapat disebabkan oleh bakteri, jamur, atau virus. 
        Penyakit ini menyebabkan bintik-bintik pada daun tanaman. '''
    
    elif nama == 'Powdery-Mildew':
        return '''Powdery mildew merupakan penyakit tanaman yang disebabkan oleh berbagai spesies jamur dari keluarga Erysiphaceae. 
        Penyakit ini muncul pada daun yang sedang bertunas dan membentuk lepuh yang menyebabkan daun menggulung ke atas dan memperlihatkan epidermis bagian bawah.'''
    
    elif nama == 'Rust-Leafs':
        return '''Rust leaf disebabkan oleh jamur dari keluarga Pucciniales dan terdapat >4000 spesies Rust Leaf.
        Lesi pertama kali muncul sebagai bintik putih dan sedikit menonjol pada epidermis bawah daun tanaman dewasa atau tua.'''
    
    else:
        return 'Tanaman ini terindikasi sehat.'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# config tempat folder upload
# app.config['UPLOAD_FOLDER'] = os.path.join(app.instance_path, 'uploads')
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. buat enpoint untuk menerima file dari Front end, lalu di proses 
# misal http://localhost:5000/upload
@app.route('/test', methods = ['POST'])   
def test():   
    if request.method == 'POST':   
        f = request.files['file'] 
        # save file yang di upload ke folder 'instance' /uploads
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        # tempat pemrosesan file

        # file yang sudah di proses akan disimpan di folder 'instance' /processed_file
        # processed_file = processed_file
        # f.save(os.path.join(app.instance_path, 'processed_file', secure_filename(processed_file)))

        # respon yang akan dikirim ke Front end
        response = make_response(
            jsonify(
                {
                    'uploaded_filename': f.filename,
                    'predicted_filename': 'predicted_' + f.filename,
                    'deskripsi': 'deskripsi penyakit',
                    'nama': 'Nama Penyakit',
                    'rekomendasi': 'Rekomendasi untuk penyakit',
                    'prevention': 'Prevention untuk penyakit',   
                    'url': url_for('static', filename='assets/uploads/' + f.filename)

                } # mengirim nama file yang sudah di proses, f.filename hanya contoh, dan deskripsi
            ),
            200,
        )
        response.headers['Content-Type'] = 'application/json'
        return response

# 2. buat enpoint untuk mengakses file yang telah di proses dari Backend
# misal http://localhost:5000/processed_file/<filename>.<format>
# '/uploads/<filename>' adalah endpoint yang akan diakses oleh Front end
# '/uploads/<filename>' adalah tempat file yang sudah di proses
# filename adalah nama file yang akan diakses
# @app.route('/uploads/<filename>')
# def processed_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
