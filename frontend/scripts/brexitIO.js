/*jslint browser: true*/
/*jslint vars: true, plusplus: true, devel: true, nomen: true, indent: 4, maxerr: 50 */
/*global  $*/

$(document).ready(function () {
    'use strict';
    
    $("#predictTextBox").keyup(function (event) {//enter in textbox simulates button click
        if (event.keyCode === 13) {
            $("#predict").click();
        }
    });
    
    var animating = false;
    
    $(".info").click(function () {//slidetoggle #about div with copyrighted proprietary yoyo countermeasure code
        if (animating === false) {
            $("#i-active").fadeToggle();
            $("#about").slideToggle();
            animating = true;
            setTimeout(function () {animating = false; }, 400);
        }
    });
});

var predicting = false;

function predict() {
    'use strict';
    
    if (predicting === false) {
        var stringToPredict = $("#predictTextBox").val();//get data from textbox
        
        predicting = true;
        
        $.ajax({//send data to the flask server and get a response hopefully
            url: 'http://localhost:5000/predict/',
            type: 'POST',
            data: JSON.stringify({"stringToPredict" : stringToPredict}),//give the string
            cache: false,
            success: function (response) {//get the data back and do something with it
                $(".classifierText").stop();
                $(".classifierText").fadeOut();
                setTimeout(function () { $("#" + response).fadeIn(); }, 400);
                setTimeout(function () {predicting = false; }, 700);
            },
            error: function (error) {//if there is an error, log it
                console.log(error);
            }
        });
    }
}