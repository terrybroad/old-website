



var sample_training_instance = function() {
  // find an unloaded batch
  var bi = Math.floor(Math.random()*loaded_train_batches.length);
  var b = loaded_train_batches[bi];
  var k = Math.floor(Math.random()*num_samples_per_batch); // sample within the batch
  var n = b*num_samples_per_batch+k;

  // load more batches over time
  if(step_num%(2 * num_samples_per_batch)===0 && step_num>0) {
    for(var i=0;i<num_batches;i++) {
      if(!loaded[i]) {
        // load it
        load_data_batch(i);
        break; // okay for now
      }
    }
  }

  // fetch the appropriate row of the training image and reshape into a Vol
  var p = img_data[b].data;
  var x = new convnetjs.Vol(image_dimension,image_dimension,image_channels,0.0);
  var W = image_dimension*image_dimension;
  var j=0;
  for(var dc=0;dc<image_channels;dc++) {
    var i=0;
    for(var xc=0;xc<image_dimension;xc++) {
      for(var yc=0;yc<image_dimension;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }

  if(random_position){
    var dx = Math.floor(Math.random()*5-2);
    var dy = Math.floor(Math.random()*5-2);
    x = convnetjs.augment(x, image_dimension, dx, dy, false); //maybe change position
  }

  if(random_flip){
    x = convnetjs.augment(x, image_dimension, 0, 0, Math.random()<0.5); //maybe flip horizontally
  }

  var isval = use_validation_data && n%10===0 ? true : false;
  return {x:x, label:labels[n], isval:isval};
}
// sample a random testing instance
var sample_test_instance = function() {

  var b = test_batch;
  var k = Math.floor(Math.random()*num_samples_per_batch);
  var n = b*num_samples_per_batch+k;

  var p = img_data[b].data;
  var x = new convnetjs.Vol(image_dimension,image_dimension,image_channels,0.0);
  var W = image_dimension*image_dimension;
  var j=0;
  for(var dc=0;dc<image_channels;dc++) {
    var i=0;
    for(var xc=0;xc<image_dimension;xc++) {
      for(var yc=0;yc<image_dimension;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }

  // distort position and maybe flip
  var xs = [];
  
  if (random_flip || random_position){
    for(var k=0;k<6;k++) {
      var test_variation = x;
      if(random_position){
        var dx = Math.floor(Math.random()*5-2);
        var dy = Math.floor(Math.random()*5-2);
        test_variation = convnetjs.augment(test_variation, image_dimension, dx, dy, false);
      }
      
      if(random_flip){
        test_variation = convnetjs.augment(test_variation, image_dimension, 0, 0, Math.random()<0.5); 
      }

      xs.push(test_variation);
    }
  }else{
    xs.push(x, image_dimension, 0, 0, false); // push an un-augmented copy
  }
  
  // return multiple augmentations, and we will average the network over them
  // to increase performance
  return {x:xs, label:labels[n]};
}

// int main
$(window).load(function() {
  //update_net_param_display();

  for(var k=0;k<loaded.length;k++) { loaded[k] = false; }

  load_data_batch(0); // async load train set batch 0
  load_data_batch(test_batch); // async load test set
  start_fun();
});

var load_and_step = function() {
  if(paused) return; 
  if(!visNewDigit) return;

  updateNet(); // process this image
  
  //setTimeout(load_and_step, 0); // schedule the next iteration
}
// var paused = false;
// var load_and_step = function() {
//   if(paused) return; 

//   var sample = sample_training_instance();
//   step(sample); // process this image
  
//   //setTimeout(load_and_step, 0); // schedule the next iteration
// }

var start_fun = function() {
  if(loaded[0] && loaded[test_batch]) { 
    console.log('starting!'); 
    setInterval(load_and_step, 0); // lets go!
  }
  else { setTimeout(start_fun, 200); } // keep checking
}

var reset_all = function() {
  // reinit trainer
  trainer = new convnetjs.SGDTrainer(net, {learning_rate:trainer.learning_rate, momentum:trainer.momentum, batch_size:trainer.batch_size, l2_decay:trainer.l2_decay});
  //update_net_param_display();

  // reinit windows that keep track of val/train accuracies
  // xLossWindow.reset();
  // wLossWindow.reset();
  // trainAccWindow.reset();
  // valAccWindow.reset();
  // testAccWindow.reset();
  // lossGraph = new cnnvis.Graph(); // reinit graph too
  step_num = 0;
}

var load_pretrained = function() {
  $.getJSON("network.json", function(json){
    net = new convnetjs.Net();
    net.fromJSON(json);
    trainer.learning_rate = 0.0001;
    trainer.momentum = 0.9;
    trainer.batch_size = 2;
    trainer.l2_decay = 0.00001;
    reset_all();
  });
}

var load_data_batch = function(batch_num) {
  // Load the dataset with JS in background
  data_img_elts[batch_num] = new Image();
  var data_img_elt = data_img_elts[batch_num];
  data_img_elt.onload = function() { 
    var data_canvas = document.createElement('canvas');
    data_canvas.width = data_img_elt.width;
    data_canvas.height = data_img_elt.height;
    var data_ctx = data_canvas.getContext("2d");
    data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
    img_data[batch_num] = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
    loaded[batch_num] = true;
    if(batch_num < test_batch) { loaded_train_batches.push(batch_num); }
    console.log('finished loading data batch ' + batch_num);
  };
  data_img_elt.src = dataset_name + "/" + dataset_name + "_batch_" + batch_num + ".png";
}


var test_predict = function() {
  var num_classes = net.layers[net.layers.length-1].out_depth;

  document.getElementById('testset_acc').innerHTML = '';
  var num_total = 0;
  var num_correct = 0;

  // grab a random test image
  for(num=0;num<4;num++) {
    var sample = sample_test_instance();
    var y = sample.label;  // ground truth label

    // forward prop it through the network
    var aavg = new convnetjs.Vol(1,1,num_classes,0.0);
    // ensures we always have a list, regardless if above returns single item or list
    var xs = [].concat(sample.x);
    var n = xs.length;
    for(var i=0;i<n;i++) {
      var a = net.forward(xs[i]);
      aavg.addFrom(a);
    }
    var preds = [];
    for(var k=0;k<aavg.w.length;k++) { preds.push({k:k,p:aavg.w[k]}); }
    preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
    
    var correct = preds[0].k===y;
    if(correct) num_correct++;
    num_total++;

    var div = document.createElement('div');
    div.className = 'testdiv';

    // draw the image into a canvas
    draw_activations_COLOR(div, xs[0], 2); // draw Vol into canv

    // add predictions
    var probsdiv = document.createElement('div');
    
    var t = '';
    for(var k=0;k<3;k++) {
      var col = preds[k].k===y ? 'rgb(85,187,85)' : 'rgb(187,85,85)';
      t += '<div class=\"pp\" style=\"width:' + Math.floor(preds[k].p/n*100) + 'px; background-color:' + col + ';\">' + classes_txt[preds[k].k] + '</div>'
    }
    probsdiv.innerHTML = t;
    probsdiv.className = 'probsdiv';
    div.appendChild(probsdiv);

    // add it into DOM
    $(div).prependTo($("#testset_vis")).hide().fadeIn('slow').slideDown('slow');
    if($(".probsdiv").length>200) {
      $("#testset_vis > .probsdiv").last().remove(); // pop to keep upper bound of shown items
    }
  }
  testAccWindow.add(num_correct/num_total);
  $("#testset_acc").text('test accuracy based on last 200 test images: ' + testAccWindow.get_average());  
}

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;


