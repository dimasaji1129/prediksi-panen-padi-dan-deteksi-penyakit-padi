// script.js

function previewImage(input) {
    var preview = document.getElementById('image-preview');
    var file = input.files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function showExplanation(result_label) {
    var explanation = document.getElementById('result_explanation');
    var explanations = {
        'BLB':' Bacterial Leaf Blight, penyakit yang disebabkan oleh bakteri Xanthomonas oryzae pv. oryzae pada tanaman padi. Penyakit ini mempengaruhi daun tanaman padi dan menyebabkan kerugian hasil panen',
        'BPH': 'Brown Planthopper, serangga yang menyerang tanaman padi dan dapat menyebabkan kerusakan yang signifikan. Serangga ini dapat menularkan penyakit virus dan menyebabkan kematian tanaman pada infestasi yang parah',
        'Brown_Spot': 'Penyakit daun coklat, penyakit yang disebabkan oleh jamur Cochliobolus miyabeanus pada tanaman padi. Penyakit ini mempengaruhi daun tanaman padi dan menyebabkan bercak coklat kecil dengan halo kuning',
        'False_Smut': 'Penyakit busuk palsu, penyakit yang disebabkan oleh jamur Ustilaginoidea virens pada tanaman padi. Penyakit ini mempengaruhi biji padi dan menyebabkan biji tersebut berubah menjadi bola busuk besar berwarna hijau kecoklatan',
        'Healthy_Plant': 'Tanaman sehat, tidak terinfeksi oleh penyakit atau hama',
        'Hispa': 'Serangga Hispa, serangga yang menyerang tanaman padi dan dapat menyebabkan kerusakan pada daun. Serangga ini dapat menyebabkan pola makan "jendela" khas dan pengeringan daun. Infestasi yang berat dapat menyebabkan kerugian hasil panen',
        'eck_Blast': 'Penyakit leher blast, penyakit yang disebabkan oleh jamur Magnaporthe oryzae pada tanaman padi. Penyakit ini mempengaruhi leher dan malai tanaman padi, menyebabkan busuk leher dan kehilangan sebagian atau seluruh biji',
        'Sheath_Blight_Rot': 'Penyakit busuk sarung, penyakit yang disebabkan oleh jamur Rhizoctonia solani pada tanaman padi. Penyakit ini mempengaruhi sarung dan daun tanaman padi, menyebabkan lesi dan pembusukan',
        'Stemborer': 'Penggerek batang, serangga yang menyerang batang tanaman padi dan dapat menyebabkan kerusakan yang signifikan. Serangga ini juga dapat menularkan penyakit dan melemahkan tanaman'
    };
    explanation.innerHTML = explanations[result_label] || 'Unknown disease';
}