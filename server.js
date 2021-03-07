const express = require("express");
const bodyPareser = require("body-parser");
const { spawn } = require ("child_process");

const app = express();
app.use(bodyPareser.urlencoded({extended:true}));
app.use(express.static(__dirname + '/public'));

app.get("/", function(req,res){
    res.sendFile(__dirname +"/index.html");
})

app.post("/", function(req,res){
    var content = req.body.contentInput;
    let output="";
    const childPython = spawn('python', ['main.py', content])

    childPython.stdout.on("data", (data)=>{
        output+=data.toString();
    })

    childPython.on('close', (code)=> {
        output.match("True") ? 
        res.sendFile(__dirname +"/answer.html"):
        res.sendFile(__dirname +"/answerNo.html")
        output="";
    })

})

app.get("/homePage", function(req,res){
    res.sendFile(__dirname +"/index.html");
})

app.listen(3000, function(){
    console.log("Server has started on port 3000")
});