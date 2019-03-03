const express = require('express')
const path = require('path')
const port = process.env.PORT || 8000
const app = express()

// 通常用于加载静态资源
app.use(express.static('public'))


// 在你应用 JavaScript 文件中包含了一个 script 标签
// 的 index.html 中处理任何一个 route

app.get('/public', function (request, response){
    response.sendFile(path.resolve('public', 'index.html'))
  })

app.listen(port)
console.log("server started on port " + port)