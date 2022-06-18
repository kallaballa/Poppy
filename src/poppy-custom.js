	var croppers = {};
	var blobs = {};
	var is_mobile = false;
	function isMobile() {
		let check = false;
		(function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
		return check;
	};

	function init_poppy(containerID1, containerID2, inputFileID1, inputFileID2, imageID1, imageID2) {
		is_mobile = isMobile();
		var vh = window.innerHeight * 0.01;
		// Then we set the value in the --vh custom property to the root of the document
		document.documentElement.style.setProperty('--vh', `${vh}px`);
		if (window.FileReader) {
			function handleFileSelect(containerID, imageID, evt) {
				open_crop(evt, containerID, imageID)
			}
		} else {
			alert('This browser does not support FileReader');
		}
		document.getElementById(inputFileID1).addEventListener('change', handleFileSelect.bind(null, containerID1, imageID1), false);
		document.getElementById(inputFileID2).addEventListener('change', handleFileSelect.bind(null, containerID2, imageID2), false);
	}


	function open_crop(evt, containerID, imageID) {
		var f = evt.target.files[0];
		var imgElem = document.getElementById(imageID);
		reader = new FileReader();
		reader.onload = (function(theFile) {
			return function(e) {
				document.getElementById("crop-bg").style.display = "table-cell";
				document.getElementById(containerID).style.display = "block";

				imgElem.src = e.target.result;
				if(croppers[imgElem.id] !== undefined) {
					croppers[imgElem.id].destroy();
				}

				croppers[imgElem.id] = new Cropper(imgElem, {
					autoCropArea: 0.8
				});
			};
		})(f);
		reader.readAsDataURL(f);
	}

	function make_blob(imageID, imageContainerID, targetID, index) {
		document.getElementById(imageContainerID).style.display='none';
		document.getElementById('crop-bg').style.display = 'none';
		croppers[imageID].crop();
		var croppedCanvas = croppers[imageID].getCroppedCanvas();
		var dataUrl = croppedCanvas.toDataURL('image/png');

		var maxWidth = is_mobile ? 768 : 2048;
		var maxHeight = is_mobile ? 768 : 2048;
		var srcWidth = croppedCanvas.width;
		var srcHeight = croppedCanvas.height;
		var ratio = Math.min(maxWidth / srcWidth, maxHeight / srcHeight);
		var w = srcWidth * ratio;
		var h = srcHeight * ratio;

		if(srcWidth > maxWidth || srcHeight > maxHeight) {
			alert("The image will be scaled down because of program limitiations.");
			var offCanvas = new OffscreenCanvas(w, h);
			var ctx = offCanvas.getContext("2d");
			ctx.drawImage(croppedCanvas, 0, 0, w, h);
			offCanvas.convertToBlob().then(blob => {
				blobs[index] = blob;
			});
		} else {
			croppers[imageID].getCroppedCanvas().toBlob(function(blob) {
				blobs[index] = blob;
			});
		}
		document.getElementById(targetID).src = dataUrl;
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
		var dataString = 'data:image/gif;base64,' + Uint8ToBase64(data);
		element.setAttribute('href', dataString);
		element.setAttribute('download', filename);

		element.style.display = 'none';
		document.body.appendChild(element);

		element.click();
		document.body.removeChild(element);

// 		var img = document.createElement('img');
// 		img.setAttribute('src', dataString);
// 		document.body.appendChild(img);
	}

	function scale_canvas() {
		var canvas = document.getElementById('canvas');
		var canvas = document.getElementById('canvas');
		var ctx = canvas.getContext("2d");


		var container = document.getElementById("canvasrow");
		var sx = container.clientWidth / 160;
		var sy = container.clientHeight / 160;
		ctx.scale(sx, sy);
	}

	var Module = {
		onRuntimeInitialized: function() {
			document.getElementById("content").style.visibility = "visible";
			document.getElementById("loading").style.visibility = "hidden";
 			scale_canvas();
		},
		print : (function() {
			return function(message) {
				console.log(message);
			};
		})(),
		printErr : function(message) {
			console.error(message);

			if(message.endsWith("%")) {
				document.getElementById('progress').innerHTML = message;
			}

			if(message === "done") {
				download("poppy.gif", FS.readFile("current.gif"));
				document.getElementById('canvasrow').style.display = "none";
				document.getElementById('headerrow').style.display = "table-row";
				document.getElementById('startbtn').disabled = false;
				document.getElementById('progress').innerHTML = "Initializing...";
			}
		},
		canvas : (function() {
			return document.getElementById('canvas');
		})()
	};
	window.addEventListener("resize", function(e) {
		scale_canvas();
	}, true);
	Module.doNotCaptureKeyboard = true;

	function load_images() {
		document.getElementById('canvasrow').style.display = "table-row";
		document.getElementById('headerrow').style.display = "none";
		document.getElementById('startbtn').disabled = true;

		var arrayBuffer;
		var fileReader1 = new FileReader();
		var fileReader2 = new FileReader();
		var tolerance  = document.getElementById('tolerance').value;
		var autoalign = document.getElementById('autoalign').checked;
//		var face = document.getElementById('face').checked;
		var face = false;
		var autoscale = document.getElementById('autoscale').checked;
		var numberOfFrames  = document.getElementById('frames').value;
		fileReader1.onload = function(event) {
			const uint8_view1 = new Uint8Array(event.target.result);
			FS.writeFile('temp1.png', uint8_view1)
			fileReader2.readAsArrayBuffer(blobs[1]);
			fileReader2.onload = function(event) {
				const uint8_view2 = new Uint8Array(event.target.result);
				FS.writeFile('temp2.png', uint8_view2)
				Module.ccall('load_images', 'number', [ 'string', 'string', 'number', 'boolean', 'boolean', 'number', 'bool' ], [ 'temp1.png', 'temp2.png', tolerance, face, autoscale, numberOfFrames, autoalign ])
				for(var key in croppers) {
					var c = croppers[key];
					c.destroy();
				}

				document.getElementById("image1file").value = "";
				document.getElementById("image2file").value = "";
				document.getElementById("canvas").getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
			};
		};
		fileReader1.readAsArrayBuffer(blobs[0]);
	}
