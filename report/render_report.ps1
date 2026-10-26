$ErrorActionPreference = "Stop"
$d = "E:\BTech 3rd Year (Sem 5)\PCAP-Project\report\PCAP_Report_IEEE.docx"
$pdf = "E:\BTech 3rd Year (Sem 5)\PCAP-Project\report\PCAP_Report_IEEE.pdf"
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
  # Open(FileName, ConfirmConversions, ReadOnly, AddToRecentFiles, PasswordDocument, PasswordTemplate, Revert, WritePasswordDocument, WritePasswordTemplate, Format, Encoding, Visible, OpenAndRepair, DocumentDirection, NoEncodingDialog)
  $doc = $w.Documents.Open($d, $false, $true, $false, "", "", $true, "", "", 0, 0, $false, $false, 0, $true)
  $doc.ExportAsFixedFormat($pdf, 17)
  "pages: " + $doc.ComputeStatistics(2)
  $doc.Close(0)
} finally { $w.Quit() }
