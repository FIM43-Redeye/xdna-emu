function insertFooter() {
  var fileName = location.pathname.substring(location.pathname.lastIndexOf("/") + 1);
  fileName = fileName.substring(0,fileName.lastIndexOf("."));
  var footerText="<div><span style=float:left;padding-left:6pt>AM025 (v1.1) November 13, 2024&nbsp;&nbsp;&#169; Copyright 2022-2024 Advanced Micro Devices, Inc. All rights reserved.&nbsp;&nbsp;&nbsp;&nbsp;<a href=_overview.html#inclusive>Inclusive Terminology</a></span><span style=float:right;text-align:right;padding-right:4pt><a href=https://www.xilinx.com/about/feedback/document-feedback.html?docType=User_Guides&docId=AM025&Title=Versal%20Adaptive%20SoC%20AIE-ML%20Register%20Reference&releaseVersion=1.1&docPage=" + fileName + " target=_blank><img src=feedback_icon_true_size_e.png height=23px width=87px border=0></a></span></div>";
  document.getElementById("foot").innerHTML=footerText;
}
window.onload=insertFooter;

function gotoTopic(thisTopic) {
  if (window.top != window.self) {
    window.self.location.href = thisTopic;
  } else {
    window.top.location.href="./am025-versal-aie-ml-register-reference.html#" + thisTopic;
  }
  return false;
}
