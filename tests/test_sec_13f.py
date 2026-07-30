"""Regression tests for whaletrading.data.sec_13f XML parsing.

Fixtures below reproduce the exact root-element shape (minus the bulk of
real holdings) of two filings that failed in production with
"unbound prefix" ElementTree errors:
  - BlackRock, accession 0001086364-24-008417 (fails at line 3)
  - T. Rowe Price, accession 0000080255-26-000381 (fails at line 2)

Both declare an `xsi:schemaLocation="..."` attribute on the root element.
_parse_info_table's namespace-stripping regex used to strip the
`xmlns:xsi=` declaration but leave that attribute's `xsi:` prefix behind,
which ElementTree rejects as unbound.
"""

from __future__ import annotations

from whaletrading.data.sec_13f import _parse_info_table

# Shape of BlackRock's form13fInfoTable.xml: xsi:schemaLocation appears
# after the xmlns declarations.
BLACKROCK_SHAPE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<?xml-stylesheet type='text/xsl' href="INFO-TABLE_X01.xsl"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:ns2="http://www.sec.gov/edgar/common" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sec.gov/edgar/document/thirteenf/informationtable eis_13FDocument.xsd">
\t<infoTable>
\t\t<nameOfIssuer>1 800 FLOWERS COM INC</nameOfIssuer>
\t\t<titleOfClass>CL A</titleOfClass>
\t\t<cusip>68243Q106</cusip>
\t\t<value>3426648</value>
\t\t<shrsOrPrnAmt>
\t\t\t<sshPrnamt>359942</sshPrnamt>
\t\t\t<sshPrnamtType>SH</sshPrnamtType>
\t\t</shrsOrPrnAmt>
\t\t<investmentDiscretion>SOLE</investmentDiscretion>
\t</infoTable>
</informationTable>
"""

# Shape of T. Rowe Price's infotable.xml: xsi:schemaLocation appears BEFORE
# the xmlns declarations, on the first line after the XML declaration.
T_ROWE_PRICE_SHAPE = """<?xml version="1.0" ?>
<informationTable xsi:schemaLocation="http://www.sec.gov/edgar/document/thirteenf/informationtable eis_13FDocument.xsd" xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:n1="http://www.sec.gov/edgar/document/thirteenf/informationtable" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <infoTable>
    <nameOfIssuer>AAON INC</nameOfIssuer>
    <titleOfClass>COMM STK</titleOfClass>
    <cusip>000360206</cusip>
    <value>5485</value>
    <shrsOrPrnAmt>
      <sshPrnamt>66274</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>DFND</investmentDiscretion>
  </infoTable>
</informationTable>
"""

# Shape of a filer with no xsi:schemaLocation at all (the 8 managers that
# already worked before the fix) -- must keep working.
NO_XSI_SHAPE = """<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>TEST CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>123456789</cusip>
    <value>1000</value>
    <shrsOrPrnAmt><sshPrnamt>500</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>
"""


def test_parses_blackrock_shape_with_trailing_xsi_attribute():
    holdings = _parse_info_table(BLACKROCK_SHAPE)
    assert holdings == [{"issuer": "1 800 FLOWERS COM INC", "shares": 359942, "value": 3426648}]


def test_parses_t_rowe_price_shape_with_leading_xsi_attribute():
    holdings = _parse_info_table(T_ROWE_PRICE_SHAPE)
    assert holdings == [{"issuer": "AAON INC", "shares": 66274, "value": 5485}]


def test_parses_filer_shape_without_xsi_attribute():
    holdings = _parse_info_table(NO_XSI_SHAPE)
    assert holdings == [{"issuer": "TEST CORP", "shares": 500, "value": 1000}]
