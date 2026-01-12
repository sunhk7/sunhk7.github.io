document.addEventListener("DOMContentLoaded", function() {
  // 1. 选中所有 Python 代码块
  // Quarto 的代码块通常在 pre.sourceCode.python code 中
  const codeBlocks = document.querySelectorAll('pre.sourceCode.python code, code.sourceCode.python');

  codeBlocks.forEach(block => {
    let html = block.innerHTML;

    // --- 手术 1: 修复函数名 (def 之后的单词) ---
    // 原理：寻找 <span class="kw">def</span> 后面紧跟的单词
    // 这里的 kw 是 Pandoc 给 def 上的色
    html = html.replace(
      /(<span class="kw">def<\/span>\s+)(\w+)/g, 
      '$1<span class="custom-func">$2</span>'
    );

    // --- 手术 2: 修复变量名 (等号 = 之前的单词) ---
    // 原理：寻找 "单词 + 空格 + =" 的结构
    // 注意：我们要避开已经有标签的内容，只匹配纯文本的变量
    // 下面的正则匹配：单词(group1) + 可能的空格 + =
    // (?!<) 确保前面不是标签的一部分
    html = html.replace(
      /(?<!>)\b(\w+)(\s*=\s*)(?![^<]*>)/g, 
      '<span class="custom-var">$1</span>$2'
    );

    block.innerHTML = html;
  });
});