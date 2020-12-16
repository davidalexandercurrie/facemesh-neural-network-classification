let facemesh;
let video;
let predictions = [];

let model;
let targetLabel;
let state = 'collection';

let nnResults;
let loopBroken = true;

let inp;
let submit;

let nameValue = '';

let socket;

let model1ready = false;
let model2ready = false;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  socket = io.connect();

  facemesh = ml5.facemesh(video, modelReady);

  let options = {
    inputs: 1404,
    outputs: ['label'],
    task: 'classification',
    debug: 'false',
  };

  model = ml5.neuralNetwork(options);
  // autoStartPredict();

  // This sets up an event that fills the global variable "predictions"
  // with an array every time new predictions are made
  facemesh.on('predict', results => {
    predictions = results;
    if (!model1ready) {
      model1ready = true;
      onPredictClick();
    }
  });

  // Hide the video element, and just show the canvas
  video.hide();

  createButton('Load Model').mousePressed(onLoadModelClick);
  createButton('Start Prediction').mousePressed(onPredictClick);
}

function autoStartPredict() {
  if (state == 'prediction') {
    onLoadModelClick();
    onPredictClick();
  }
}

function dataLoaded() {
  console.log(model.data);
}

function modelReady() {
  console.log('Model ready!');
}

function draw() {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints
  drawKeypoints();
  restartPredictions();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const keypoints = predictions[i].scaledMesh;

    // Draw facial keypoints.
    for (let j = 0; j < keypoints.length; j += 1) {
      const [x, y] = keypoints[j];

      fill(0, 255, 0);
      ellipse(x, y, 5, 5);
    }
  }
}

function mousePressed() {
  if (predictions[0] != undefined) {
    let inputs = predictions[0].mesh.flat();
    if (state == 'collection') {
      let target = {
        label: targetLabel,
      };
      if (targetLabel != undefined) {
        model.addData(inputs, target);
        console.log(`Data recorded for label ${targetLabel}`);
      } else {
        console.log('Target label not set.');
      }
      text(targetLabel, mouseX, mouseY);
    } else if (state == 'prediction') {
      model.classify(inputs, gotResults);
    }
  }
}

function gotResults(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  console.log(`${results[0].label}: ${results[0].confidence}`); // print label & confidence
  nnResults = results;
  sendToServer();
  classify();
  if (!model2ready) {
    inp = createInput('name');
    inp.input(myInputEvent);
    submitButton = createButton('submit');
    submitButton.mousePressed(sendName);
    model2ready = true;
  }
}

function myInputEvent() {}

function keyPressed() {
  if (key == 't') {
    console.log('starting training');
    state = 'training';
    model.normalizeData();
    let options = {
      epochs: 50,
    };
    model.train(options, whileTraining, finishedTraining);
  } else if (key == 's') {
    model.saveData();
  } else if (key == 'm') {
    model.save();
  } else {
    targetLabel = key.toUpperCase();
  }
}

function whileTraining(epoch, loss) {
  console.log(epoch, loss);
}
function finishedTraining() {
  console.log('finished training');
}

function classify() {
  if (predictions[0] != undefined) {
    let inputs = predictions[0].mesh.flat();
    model.classify(inputs, gotResults);
  } else {
    loopBroken = true;
  }
}

function onPredictClick() {
  state = 'prediction';
}
function onLoadModelClick() {
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  model.load(modelInfo, () => console.log('Model Loaded.'));
}

function restartPredictions() {
  if (loopBroken) {
    loopBroken = false;
    classify();
  }
}

const sendToServer = () => {
  let data = {
    data: nnResults,
    name: nameValue,
  };
  console.log(data);
  socket.emit('facemesh', nnResults);
};

function sendName() {
  nameValue = inp.value();
  inp.value('');
}
