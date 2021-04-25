const fs = require("fs")
const path = process.argv[2]
const f = JSON.parse(fs.readFileSync(path, "utf8"))
const out = f.cells
    .map((x, i) => {
        const cleaned = x.source.map(x => {
            if (x.startsWith("!pip")) {
                return `# ${x}`
            } else {
                return x
            }
        }).join("")
        return `# [${i}] START\n${cleaned}\n# [${i}] END\n`
    })
    .reduce((acc, cur) => [...acc, cur], [])
    .join("\n")
console.log(out)
