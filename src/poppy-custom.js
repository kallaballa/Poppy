	var croppers = {};

	function crop(fileBlob, imgElem) {
		if(document.getElementById('imageload1').value !== "" && document.getElementById('imageload2').value != "") {
			document.getElementById('startbtn').disabled = false;
		}
		reader = new FileReader();
		reader.readAsArrayBuffer(fileBlob);
		reader.addEventListener('loadend', function(e) {
			let result = reader.result;
			const uint8_view = new Uint8Array(result);
			imgElem.src = 'data:image/png;base64,' + Uint8ToBase64(uint8_view);
			if(croppers[imgElem.id] !== undefined) {
				croppers[imgElem.id].destroy();
			}

			croppers[imgElem.id] = new Cropper(imgElem, {
					// The view mode of the cropper
				viewMode: 2, // 0, 1, 2, 3
				// The dragging mode of the cropper
				dragMode: 'crop', // 'crop', 'move' or 'none'
				// The initial aspect ratio of the crop box
				initialAspectRatio: NaN,
				// The aspect ratio of the crop box
				aspectRatio: NaN,
				// An object with the previous cropping result data
				data: null,
				// A selector for adding extra containers to preview
				preview: '',
				// Re-render the cropper when resize the window
				responsive: true,
				// Restore the cropped area after resize the window
				restore: true,
				// Check if the current image is a cross-origin image
				checkCrossOrigin: true,
				// Check the current image's Exif Orientation information
				checkOrientation: true,
				// Show the black modal
				modal: true,
				// Show the dashed lines for guiding
				guides: true,
				// Show the center indicator for guiding
				center: true,
				// Show the white modal to highlight the crop box
				highlight: true,
				// Show the grid background
				background: true,
				// Enable to crop the image automatically when initialize
				autoCrop: true,
				// Define the percentage of automatic cropping area when initializes
				autoCropArea: 0.8,
				// Enable to move the image
				movable: true,
				// Enable to rotate the image
				rotatable: true,
				// Enable to scale the image
				scalable: true,
				// Enable to zoom the image
				zoomable: true,
				// Enable to zoom the image by dragging touch
				zoomOnTouch: true,
				// Enable to zoom the image by wheeling mouse
				zoomOnWheel: true,
				// Define zoom ratio when zoom the image by wheeling mouse
				wheelZoomRatio: 0.1,
				// Enable to move the crop box
				cropBoxMovable: false,
				// Enable to resize the crop box
				cropBoxResizable: false,
				// Toggle drag mode between "crop" and "move" when click twice on the cropper
				toggleDragModeOnDblclick: true,
				// Size limitation
				minCanvasWidth: 0,
				minCanvasHeight: 0,
				minCropBoxWidth: 0,
				minCropBoxHeight: 0,
				minContainerWidth: 200,
				minContainerHeight: 100,
				// Shortcuts of events
				ready: null,
				cropstart: null,
				cropmove: null,
				cropend: null,
				crop: null,
				zoom: null
			});
		});
	}

	function Uint8ToBase64(u8Arr){
		var CHUNK_SIZE = 0x8000; //arbitrary number
		var index = 0;
		var length = u8Arr.length;
		var result = '';
		var slice;
		while (index < length) {
			slice = u8Arr.subarray(index, Math.min(index + CHUNK_SIZE, length));
			result += String.fromCharCode.apply(null, slice);
			index += CHUNK_SIZE;
		}
		return btoa(result);
	}
	function download(filename, data) {
		var element = document.createElement('a');
		element.setAttribute('href', 'data:image/gif;base64,' + Uint8ToBase64(data));
		element.setAttribute('download', filename);

		element.style.display = 'none';
		document.body.appendChild(element);

		element.click();

		document.body.removeChild(element);
	}

	function resize_canvas() {
		var canvas = document.getElementById('canvas');
		var container = document.getElementById("content");
		var sx = canvas.style.width / container.clientWidth;
		var sy = canvas.style.height / container.clientHeight;
		if(sy < sx) {
			canvas.style.width = container.clientWidth + "px";
			canvas.style.height = (canvas.style.height * sx) + "px";
		} else {
			canvas.style.width = (canvas.style.width * sy) + "px";
			canvas.style.height = container.clientHeight + "px";
		}
	}

	var Module = {
		onRuntimeInitialized: function() {
			document.getElementById("content").style.visibility = "visible";
			document.getElementById("loading").style.visibility = "hidden";
			resize_canvas();
		},
		print : (function() {
			return function(message) {
				out.innerHTML += message + '\n';
				console.log(message);
			};
		})(),
		printErr : function(message) {
			stderr.innerHTML += '> ' + message + '\n';
			stderr.scrollTop = stderr.scrollHeight;
			console.error(message);
			if(message === "done") {
				download("poppy.gif", FS.readFile("current.gif"));
				document.getElementById('startbtn').disabled = false;
			}
		},
		canvas : (function() {
			return document.getElementById('canvas');
		})()
	};
	window.addEventListener("resize", function(e) {
		resize_canvas();
	}, true);
	Module.doNotCaptureKeyboard = true;
	var debugBtn = document.getElementById("debugBtn");
	debugBtn.onclick = function() {
		var err = document.getElementById("stderr");
		var out = document.getElementById("stdout");
		if(err.style.visibility === undefined || err.style.visibility === "" || err.style.visibility === "hidden") {
			out.style.visibility = "visible";
			err.style.visibility = "visible";
		} else {
			out.style.visibility = "hidden";
			err.style.visibility = "hidden";
		}
	}
	let reader1;
	let reader2;

	function load_files() {
		document.getElementById('startbtn').disabled = true;
		reader1 = new FileReader();
		reader2 = new FileReader();
		let file1 = document.getElementById('imageload1').files[0];
		let file2 = document.getElementById('imageload2').files[0];

		reader1.readAsArrayBuffer(file1);
		reader1.addEventListener('loadend', function(e) {
			reader2.addEventListener('loadend', load_images);
			reader2.readAsArrayBuffer(file2);
		});
	}

	function load_images(e) {
		let result1 = reader1.result;
		const uint8_view1 = new Uint8Array(result1);
		let result2 = reader2.result;
		const uint8_view2 = new Uint8Array(result2);
		let tolerance  = document.getElementById('tolerance').value;
		let face  = document.getElementById('face').value === "on";
		FS.writeFile('temp1.png', uint8_view1)
		FS.writeFile('temp2.png', uint8_view2)

		Module.ccall('load_images', 'number', [ 'string', 'string', 'number', 'boolean' ], [
				'temp1.png', 'temp2.png', tolerance, face ])
		for(var key in croppers) {
			var c = croppers[key];
			c.destroy();
		}
	}
