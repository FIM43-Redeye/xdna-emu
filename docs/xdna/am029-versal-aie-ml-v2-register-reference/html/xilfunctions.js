function insertFooter() {
  var fileName = location.pathname.substring(location.pathname.lastIndexOf("/") + 1);
  fileName = fileName.substring(0,fileName.lastIndexOf("."));
  var footerText="<div><span style=float:right;padding-right:4pt><img src=feedback_icon_true_size_e.png height=23px width=87px border=0></a></span><span style=padding-left:6pt>AM029 (v1.0) May 7, 2025&nbsp;&nbsp;&#169; Copyright 2025 Advanced Micro Devices, Inc. All rights reserved.<span style=float:right;text-align:right;padding-right:4pt><a href=https://www.xilinx.com/about/feedback/document-feedback.html?docType=User_Guides&docId=AM029&Title=Versal%20Adaptive%20SoC%20AI%20Engine-ML%20v2%20Register%20Reference&releaseVersion=1.0&docPage=" + fileName + " target=_blank></span></div>";
  document.getElementById("foot").innerHTML=footerText;
}
window.onload=insertFooter;

function gotoTopic(thisTopic) {
  if (window.top != window.self) {
    window.self.location.href = thisTopic;
  } else {
    window.top.location.href="./am029-versal-aie-ml-v2-register-reference.html#" + thisTopic;
  }
  return false;
}
