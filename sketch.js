let facemesh;
let video;
let predictions = [];

let model;
let targetLabel;
let state = 'collection';

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  facemesh = ml5.facemesh(video, modelReady);

  let options = {
    inputs: 1404,
    outputs: ['label'],
    task: 'classification',
    debug: 'true',
  };

  model = ml5.neuralNetwork(options);

  // This sets up an event that fills the global variable "predictions"
  // with an array every time new predictions are made
  facemesh.on('predict', results => {
    predictions = results;
  });

  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  console.log('Model ready!');
}

function draw() {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints
  drawKeypoints();
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
    console.log(predictions[0]);
    let inputs = predictions[0].mesh.flat();
    if (state == 'collection') {
      let target = {
        label: targetLabel,
      };
      model.addData(inputs, target);
      console.log('data recorded', inputs, target);
    }
  }
}

function keyPressed() {
  console.log('starting training');
  if (key == 't') {
    model.normalizeData();
    let options = {
      epochs: 400,
    };

    model.train(options, whileTraining, finishedTraining);
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
