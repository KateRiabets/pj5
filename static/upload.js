document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('videoFile');
    const fileLabel = document.getElementById('fileLabel');
    const processBtn = document.getElementById('processBtn');
    const fileNameSpan = document.getElementById('fileName');
    const loadingDiv = document.getElementById('loading');
    const progressDiv = document.getElementById('progress');
    const downloadLinkDiv = document.getElementById('downloadLink');

    fileInput.onchange = function(e) {
        fileNameSpan.textContent = e.target.files[0].name;
        fileNameSpan.classList.remove('hidden');
        fileLabel.classList.add('hidden');
        processBtn.classList.remove('hidden');
    };

    document.getElementById('uploadForm').onsubmit = function(e) {
    e.preventDefault(); // Предотвратить стандартную отправку формы

    const formData = new FormData();
    const videoFile = fileInput.files[0];
    formData.append('video', videoFile);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);

    xhr.onloadstart = function() {
        document.getElementById('uploadForm').classList.add('hidden');
        loadingDiv.classList.remove('hidden');
        progressDiv.textContent = 'Ваше видео обрабатывается... Пожалуйста, подождите.';
    };

    xhr.onload = function() {
        if (this.status == 202) {
            const uploadedFilename = 'processed_' + videoFile.name; // Предполагаемое имя обработанного файла
            checkProcessingStatus(uploadedFilename);
        } else {
            progressDiv.textContent = 'Произошла ошибка при загрузке файла.';
        }
    };

    xhr.send(formData);
};


function checkProcessingStatus(filename) {
    const checkStatusInterval = setInterval(() => {
        fetch(`/status/${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'done') {
                    clearInterval(checkStatusInterval);
                    loadingDiv.classList.add('hidden');
                    downloadLinkDiv.innerHTML = `Обрабку завершено. <a href="/downloads/${filename}" class="btn">Завантажити відео</a>`;
                    downloadLinkDiv.style.display = 'flex'; // Показываем элемент с ссылкой для скачивания, устанавливая display в 'flex'
                } else if (data.status === 'not_found') {
                    clearInterval(checkStatusInterval);
                    progressDiv.textContent = 'Файл не найден.';
                } else {
                    progressDiv.textContent = `Обработка: ${data.status}`;
                }
            })
            .catch(error => {
                console.error('Ошибка при проверке статуса:', error);
                clearInterval(checkStatusInterval);
                progressDiv.textContent = 'Произошла ошибка при проверке статуса.';
            });
    }, 2000);
}
});
