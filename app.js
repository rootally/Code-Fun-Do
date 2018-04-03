var express = require('express');
const fileUpload = require('express-fileupload');
var path = require('path');
var favicon = require('serve-favicon');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');

var request = require('request');

var index = require('./routes/index');
var users = require('./routes/users');

var app = express();

var first = true;

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

app.use(fileUpload());

// uncomment after placing your favicon in /public
//app.use(favicon(path.join(__dirname, 'public', 'favicon.ico')));
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', index);
app.use('/users', users);

app.post('/upload', function(req, res) {

    if (!req.files)
        return res.status(400).send('No files were uploaded.');

    // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
    var sampleFile = req.files.sampleFile;
    //console.log(sampleFile);

    //Use the mv() method to place the file somewhere on your server
    sampleFile.mv('/home/bennyhawk/Desktop/IENC/test.jpg', function(err) {
        if (err){
            console.log(err);
            return res.status(500).send(err);
        }
    });
    
    console.log("Saved test.jpg");

    if(first) {
	first = false;
	return res.json({data: 'This is Smart I. You are being connected to the server.'});
    }

    request.post(
        'http://127.0.0.1:7990/get_caption', {
        form: {
            path: '/home/bennyhawk/Desktop/IENC/test.jpg'
        }
    },
    (error, response, body) => {
	res.json({
            data: body.substring(1, body.length-2)
        });
        console.log(body);
    });

});

app.post('/upload1', function(req, res) {

    if (!req.files)
        return res.status(400).send('No files were uploaded.');

    // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
    var sampleFile = req.files.sampleFile;
    //console.log(sampleFile);

    //Use the mv() method to place the file somewhere on your server
    sampleFile.mv('/home/bennyhawk/Desktop/IENC/test1.jpg', function(err) {
        if (err){
            console.log(err);
            return res.status(500).send(err);
        }
    });

    console.log("Saved test1.jpg");

    if(first) {
        first = false;
        return res.json({data: 'This is Smart I. You are being connected to the server.'});
    }    

    request.post(
        'http://127.0.0.1:7980/get_depth', {
        form: {
            path: '/home/bennyhawk/Desktop/IENC/test1.jpg'
        }
    },
    (error, response, body) => {
	res.json({
            data: body
        });
        console.log(body);
    });

});

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;

