#!/usr/bin/env python3
"""
Invoice Exception Resolution Agent
Automated detection, classification, and resolution recommendation for SAP invoice exceptions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/noirxvii/Downloads/archive (1)")

class ExceptionSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class ExceptionCategory(Enum):
    PAYMENT_BLOCK = "PAYMENT_BLOCK"
    PRICE_VARIANCE = "PRICE_VARIANCE"
    QUANTITY_VARIANCE = "QUANTITY_VARIANCE"
    THREE_WAY_MISMATCH = "THREE_WAY_MISMATCH"
    OVERDUE = "OVERDUE"
    DUPLICATE = "DUPLICATE"
    MISSING_MASTER_DATA = "MISSING_MASTER_DATA"
    TAX_ISSUE = "TAX_ISSUE"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    VENDOR_BLOCK = "VENDOR_BLOCK"

@dataclass
class InvoiceException:
    exception_id: str
    category: ExceptionCategory
    severity: ExceptionSeverity
    description: str
    affected_documents: List[str]
    vendor_id: str
    company_code: str
    amount: float
    currency: str
    root_cause: str
    recommended_action: str
    auto_resolvable: bool
    confidence_score: float
    metadata: Dict = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

class DataLoader:
    """Efficient data loading with caching"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache = {}
    
    def load(self, table: str, nrows: Optional[int] = None, 
             usecols: Optional[List[str]] = None) -> pd.DataFrame:
        key = f"{table}_{nrows}_{usecols}"
        if key in self._cache:
            return self._cache[key].copy()
        
        filepath = self.data_dir / f"{table.lower()}.csv"
        if not filepath.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, nrows=nrows, usecols=usecols, low_memory=False)
        self._cache[key] = df.copy()
        return df
    
    def load_full(self, table: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        return self.load(table, nrows=None, usecols=usecols)

class InvoiceExceptionDetector:
    """Main exception detection engine"""
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.exceptions: List[InvoiceException] = []
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load master data for reference"""
        print("Loading reference data...")
        self.lfa1 = self.loader.load('lfa1', usecols=['lifnr', 'land1', 'ktokk', 'sperr', 'term_li', 'name1'])
        self.kna1 = self.loader.load('kna1', usecols=['kunnr', 'land1', 'ktokd', 'sperr', 'name1'])
        
        # Normalize vendor/customer IDs
        for df, col in [(self.lfa1, 'lifnr'), (self.kna1, 'kunnr')]:
            if not df.empty and col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.zfill(10)
    
    def detect_all(self, sample_size: int = 100000) -> List[InvoiceException]:
        """Run all detection rules"""
        print(f"Running exception detection on {sample_size} records...")
        
        # Load transactional data
        rbkp = self.loader.load('rbkp', nrows=sample_size, 
            usecols=['mandt', 'belnr', 'gjahr', 'blart', 'bldat', 'budat', 'lifnr', 
                     'waers', 'rmwwr', 'zterm', 'xrech', 'rbstat', 'xblnr', 'bukrs',
                     'reindat', 'xmwst', 'mwskz1', 'mwskz2', 'wmwst1', 'wmwst2',
                     'name1', 'ort01', 'land1'])
        
        bseg = self.loader.load('bseg', nrows=sample_size,
            usecols=['bukrs', 'belnr', 'gjahr', 'buzei', 'shkzg', 'dmbtr', 'wrbtr',
                     'mwskz', 'saknr', 'hkont', 'kunnr', 'lifnr', 'ebeln', 'ebelp',
                     'matnr', 'werks', 'menge', 'meins', 'augdt', 'augbl', 'koart'])
        
        ekko = self.loader.load('ekko', nrows=50000,
            usecols=['ebeln', 'bukrs', 'lifnr', 'waers', 'zterm', 'bsart', 'ekorg'])
        
        ekpo = self.loader.load('ekpo', nrows=100000,
            usecols=['ebeln', 'ebelp', 'netpr', 'peinh', 'menge', 'meins', 'matnr',
                     'werks', 'pstyp', 'mwskz'])
        
        # Normalize IDs
        for df, col in [(rbkp, 'lifnr'), (bseg, 'lifnr'), (bseg, 'kunnr'), 
                        (ekko, 'lifnr')]:
            if not df.empty and col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.zfill(10)
        
        if not bseg.empty and 'ebelp' in bseg.columns:
            bseg['ebelp'] = bseg['ebelp'].astype(str).str.strip().str.zfill(5)
        if not ekpo.empty and 'ebelp' in ekpo.columns:
            ekpo['ebelp'] = ekpo['ebelp'].astype(str).str.strip().str.zfill(5)
        
        # Run detection rules
        self._detect_payment_blocks(rbkp)
        self._detect_overdue_invoices(rbkp)
        self._detect_duplicates(rbkp)
        self._detect_price_variance(bseg, ekpo)
        self._detect_quantity_variance(bseg, ekpo)
        self._detect_three_way_mismatch(rbkp, bseg, ekko, ekpo)
        self._detect_vendor_blocks(rbkp)
        self._detect_tax_issues(rbkp, bseg)
        self._detect_currency_mismatch(rbkp, ekko)
        self._detect_missing_master_data(rbkp)
        
        print(f"Total exceptions detected: {len(self.exceptions)}")
        return self.exceptions
    
    def _detect_payment_blocks(self, rbkp: pd.DataFrame):
        """Detect invoices with payment blocks"""
        if rbkp.empty or 'xrech' not in rbkp.columns:
            return
        
        blocked = rbkp[rbkp['xrech'] == 'X'].copy()
        if blocked.empty:
            return
        
        blocked['rmwwr_num'] = pd.to_numeric(blocked['rmwwr'], errors='coerce')
        
        for _, row in blocked.iterrows():
            exc = InvoiceException(
                exception_id=f"PAYBLOCK_{row['belnr']}_{row['gjahr']}",
                category=ExceptionCategory.PAYMENT_BLOCK,
                severity=ExceptionSeverity.HIGH,
                description=f"Invoice {row['belnr']}/{row['gjahr']} has payment block (XRECH=X)",
                affected_documents=[f"{row['belnr']}/{row['gjahr']}"],
                vendor_id=str(row.get('lifnr', '')),
                company_code=str(row.get('bukrs', '')),
                amount=float(row.get('rmwwr_num', 0)),
                currency=str(row.get('waers', '')),
                root_cause="Payment block set on invoice - requires manual release or resolution of underlying issue",
                recommended_action="Review invoice for discrepancies, check vendor master for payment blocks, verify goods receipt completion, then release payment block via transaction MRBR",
                auto_resolvable=False,
                confidence_score=0.95,
                metadata={
                    'blart': str(row.get('blart', '')),
                    'bldat': str(row.get('bldat', '')),
                    'zterm': str(row.get('zterm', '')),
                    'rbstat': str(row.get('rbstat', '')),
                }
            )
            self.exceptions.append(exc)
    
    def _detect_overdue_invoices(self, rbkp: pd.DataFrame):
        """Detect overdue invoices based on payment terms"""
        if rbkp.empty or 'bldat' not in rbkp.columns:
            return
        
        rbkp = rbkp.copy()
        rbkp['bldat_dt'] = pd.to_datetime(rbkp['bldat'], errors='coerce')
        rbkp['reindat_dt'] = pd.to_datetime(rbkp['reindat'], errors='coerce')
        rbkp['rmwwr_num'] = pd.to_numeric(rbkp['rmwwr'], errors='coerce')
        
        # Use receipt date if available, else document date
        rbkp['effective_date'] = rbkp['reindat_dt'].fillna(rbkp['bldat_dt'])
        
        # Parse payment terms (simplified - real implementation would use T052)
        # ZTERM examples: 0001=Immediate, 0002=14 days, 0003=30 days, etc.
        term_days = {
            '0001': 0, '0002': 14, '0003': 30, '0004': 45, '0005': 60,
            '0006': 90, '0007': 120, '0008': 180, '0009': 360,
            '1000': 30, '1001': 10, '1002': 20, '1003': 45,
            '3000': 30, '3001': 60, '3002': 90,
            'I1': 30, 'I2': 60, 'I3': 90, 'I4': 120,
        }
        
        def get_due_date(row):
            days = term_days.get(str(row.get('zterm', '')).strip(), 30)
            if pd.notna(row['effective_date']):
                return row['effective_date'] + timedelta(days=days)
            return pd.NaT
        
        rbkp['due_date'] = rbkp.apply(get_due_date, axis=1)
        rbkp['days_overdue'] = (datetime.now() - rbkp['due_date']).dt.days
        
        # Filter overdue invoices that aren't completed (RBSTAT != 'X' or similar)
        overdue = rbkp[
            (rbkp['days_overdue'] > 0) & 
            (rbkp['days_overdue'] <= 365) &  # Within a year
            (rbkp.get('rbstat', '').astype(str) != 'X')
        ].copy()
        
        for _, row in overdue.head(500).iterrows():  # Limit for performance
            severity = ExceptionSeverity.CRITICAL if row['days_overdue'] > 90 else \
                       ExceptionSeverity.HIGH if row['days_overdue'] > 30 else \
                       ExceptionSeverity.MEDIUM
            
            exc = InvoiceException(
                exception_id=f"OVERDUE_{row['belnr']}_{row['gjahr']}",
                category=ExceptionCategory.OVERDUE,
                severity=severity,
                description=f"Invoice {row['belnr']}/{row['gjahr']} overdue by {int(row['days_overdue'])} days",
                affected_documents=[f"{row['belnr']}/{row['gjahr']}"],
                vendor_id=str(row.get('lifnr', '')),
                company_code=str(row.get('bukrs', '')),
                amount=float(row.get('rmwwr_num', 0)),
                currency=str(row.get('waers', '')),
                root_cause=f"Payment terms ({row.get('zterm', 'unknown')}) exceeded. Due date: {row['due_date'].strftime('%Y-%m-%d') if pd.notna(row['due_date']) else 'unknown'}",
                recommended_action=f"Contact vendor {row.get('lifnr', '')} for payment status. Check for disputes, missing GR, or quality issues. Consider dunning process.",
                auto_resolvable=False,
                confidence_score=0.85,
                metadata={
                    'days_overdue': int(row['days_overdue']),
                    'due_date': str(row['due_date']) if pd.notna(row['due_date']) else '',
                    'zterm': str(row.get('zterm', '')),
                    'bldat': str(row.get('bldat', '')),
                }
            )
            self.exceptions.append(exc)
    
    def _detect_duplicates(self, rbkp: pd.DataFrame):
        """Detect duplicate invoice references per vendor"""
        if rbkp.empty or 'xblnr' not in rbkp.columns or 'lifnr' not in rbkp.columns:
            return
        
        rbkp = rbkp.copy()
        rbkp['xblnr_clean'] = rbkp['xblnr'].astype(str).str.strip()
        rbkp = rbkp[(rbkp['xblnr_clean'] != '') & (rbkp['xblnr_clean'] != 'nan')]
        
        dup_groups = rbkp.groupby(['lifnr', 'xblnr_clean']).filter(lambda x: len(x) > 1)
        
        for (vendor, ref), group in dup_groups.groupby(['lifnr', 'xblnr_clean']):
            docs = [f"{r['belnr']}/{r['gjahr']}" for _, r in group.iterrows()]
            amounts = [pd.to_numeric(r['rmwwr'], errors='coerce') for _, r in group.iterrows()]
            
            exc = InvoiceException(
                exception_id=f"DUP_{vendor}_{ref}",
                category=ExceptionCategory.DUPLICATE,
                severity=ExceptionSeverity.HIGH,
                description=f"Duplicate invoice reference {ref} for vendor {vendor}: {len(docs)} documents",
                affected_documents=docs,
                vendor_id=str(vendor),
                company_code=str(group.iloc[0].get('bukrs', '')),
                amount=float(sum(a for a in amounts if not pd.isna(a))),
                currency=str(group.iloc[0].get('waers', '')),
                root_cause="Same vendor invoice number used for multiple documents - possible duplicate submission or data entry error",
                recommended_action="Verify with vendor which invoice is correct. Cancel/credit duplicate. Check for automatic duplicate detection in MIRO.",
                auto_resolvable=False,
                confidence_score=0.90,
                metadata={
                    'duplicate_count': len(docs),
                    'invoice_ref': ref,
                    'individual_amounts': [float(a) for a in amounts if not pd.isna(a)],
                }
            )
            self.exceptions.append(exc)
    
    def _detect_price_variance(self, bseg: pd.DataFrame, ekpo: pd.DataFrame):
        """Detect price variance between PO and Invoice (3-way match)"""
        if bseg.empty or ekpo.empty or 'ebeln' not in bseg.columns:
            return
        
        # Filter BSEG lines with PO reference
        bseg_po = bseg[bseg['ebeln'].notna() & (bseg['ebeln'].astype(str).str.strip() != '')].copy()
        if bseg_po.empty:
            return
        
        bseg_po['ebeln'] = bseg_po['ebeln'].astype(str).str.strip().str.zfill(10)
        bseg_po['ebelp'] = bseg_po['ebelp'].astype(str).str.strip().str.zfill(5)
        bseg_po['dmbtr_num'] = pd.to_numeric(bseg_po['dmbtr'], errors='coerce')
        
        ekpo = ekpo.copy()
        ekpo['ebeln'] = ekpo['ebeln'].astype(str).str.strip().str.zfill(10)
        ekpo['ebelp'] = ekpo['ebelp'].astype(str).str.strip().str.zfill(5)
        ekpo['netpr_num'] = pd.to_numeric(ekpo['netpr'], errors='coerce')
        ekpo['peinh_num'] = pd.to_numeric(ekpo['peinh'], errors='coerce').fillna(1)
        ekpo['price_per_unit'] = ekpo['netpr_num'] / ekpo['peinh_num']
        
        # Merge on PO + Item
        merged = bseg_po.merge(
            ekpo[['ebeln', 'ebelp', 'price_per_unit', 'netpr_num', 'peinh_num', 'matnr']],
            on=['ebeln', 'ebelp'],
            how='inner'
        )
        
        if merged.empty:
            return
        
        # Calculate variance (invoice amount vs PO price * quantity)
        merged['po_amount'] = merged['price_per_unit'] * pd.to_numeric(merged['menge'], errors='coerce').fillna(1)
        merged['variance_pct'] = ((merged['dmbtr_num'] - merged['po_amount']) / merged['po_amount'].replace(0, np.nan)) * 100
        
        # Flag significant variances (>5% or >$100)
        significant = merged[
            (merged['variance_pct'].abs() > 5) | 
            ((merged['dmbtr_num'] - merged['po_amount']).abs() > 100)
        ].copy()
        
        for _, row in significant.head(200).iterrows():
            variance_amt = row['dmbtr_num'] - row['po_amount']
            severity = ExceptionSeverity.CRITICAL if abs(row['variance_pct']) > 25 else \
                       ExceptionSeverity.HIGH if abs(row['variance_pct']) > 10 else \
                       ExceptionSeverity.MEDIUM
            
            exc = InvoiceException(
                exception_id=f"PRICEVAR_{row['belnr']}_{row['buzei']}",
                category=ExceptionCategory.PRICE_VARIANCE,
                severity=severity,
                description=f"Price variance {row['variance_pct']:.1f}% on PO {row['ebeln']} item {row['ebelp']}: "
                           f"Invoice ${row['dmbtr_num']:.2f} vs PO ${row['po_amount']:.2f}",
                affected_documents=[f"{row['belnr']}/{row.get('gjahr', '')}", f"PO {row['ebeln']}/{row['ebelp']}"],
                vendor_id=str(row.get('lifnr', '')),
                company_code=str(row.get('bukrs', '')),
                amount=float(row['dmbtr_num']),
                currency='',  # Would need BKPF for currency
                root_cause="Invoice price differs from PO price - possible price change not updated, "
                          "quantity discount not applied, or wrong PO referenced",
                recommended_action="Verify with purchasing if price change was approved. Check info record (ME11/ME12) for current prices. "
                                  "If approved, update PO or accept variance. If not, request credit note from vendor.",
                auto_resolvable=False,
                confidence_score=0.85,
                metadata={
                    'po_number': str(row['ebeln']),
                    'po_item': str(row['ebelp']),
                    'invoice_amount': float(row['dmbtr_num']),
                    'po_amount': float(row['po_amount']),
                    'variance_pct': float(row['variance_pct']),
                    'variance_amount': float(variance_amt),
                    'material': str(row.get('matnr', '')),
                }
            )
            self.exceptions.append(exc)
    
    def _detect_quantity_variance(self, bseg: pd.DataFrame, ekpo: pd.DataFrame):
        """Detect quantity variance between GR and Invoice"""
        if bseg.empty or ekpo.empty or 'ebeln' not in bseg.columns:
            return
        
        bseg_po = bseg[bseg['ebeln'].notna() & (bseg['ebeln'].astype(str).str.strip() != '')].copy()
        if bseg_po.empty:
            return
        
        bseg_po['ebeln'] = bseg_po['ebeln'].astype(str).str.strip().str.zfill(10)
        bseg_po['ebelp'] = bseg_po['ebelp'].astype(str).str.strip().str.zfill(5)
        bseg_po['menge_num'] = pd.to_numeric(bseg_po['menge'], errors='coerce')
        bseg_po['dmbtr_num'] = pd.to_numeric(bseg_po['dmbtr'], errors='coerce')
        
        ekpo = ekpo.copy()
        ekpo['ebeln'] = ekpo['ebeln'].astype(str).str.strip().str.zfill(10)
        ekpo['ebelp'] = ekpo['ebelp'].astype(str).str.strip().str.zfill(5)
        ekpo['menge_num'] = pd.to_numeric(ekpo['menge'], errors='coerce')
        
        merged = bseg_po.merge(
            ekpo[['ebeln', 'ebelp', 'menge_num']].rename(columns={'menge_num': 'po_qty'}),
            on=['ebeln', 'ebelp'],
            how='inner'
        )
        
        if merged.empty:
            return
        
        merged['qty_variance'] = merged['menge_num'] - merged['po_qty']
        merged['qty_variance_pct'] = (merged['qty_variance'] / merged['po_qty'].replace(0, np.nan)) * 100
        
        significant = merged[
            (merged['qty_variance_pct'].abs() > 5) | 
            (merged['qty_variance'].abs() > 1)
        ].copy()
        
        for _, row in significant.head(200).iterrows():
            severity = ExceptionSeverity.HIGH if abs(row['qty_variance_pct']) > 20 else ExceptionSeverity.MEDIUM
            
            exc = InvoiceException(
                exception_id=f"QTYVAR_{row['belnr']}_{row['buzei']}",
                category=ExceptionCategory.QUANTITY_VARIANCE,
                severity=severity,
                description=f"Quantity variance {row['qty_variance_pct']:.1f}% on PO {row['ebeln']} item {row['ebelp']}: "
                           f"Invoice {row['menge_num']} vs PO {row['po_qty']}",
                affected_documents=[f"{row['belnr']}/{row.get('gjahr', '')}", f"PO {row['ebeln']}/{row['ebelp']}"],
                vendor_id=str(row.get('lifnr', '')),
                company_code=str(row.get('bukrs', '')),
                amount=float(row.get('dmbtr_num', 0)),
                currency='',
                root_cause="Invoiced quantity differs from ordered quantity - possible partial delivery, "
                          "over/under shipment, or wrong PO item referenced",
                recommended_action="Check goods receipt (MIGO) for actual received quantity. Verify with warehouse. "
                                  "If partial delivery, ensure GR matches actual receipt. Request credit/debit note for difference.",
                auto_resolvable=False,
                confidence_score=0.80,
                metadata={
                    'po_number': str(row['ebeln']),
                    'po_item': str(row['ebelp']),
                    'invoice_qty': float(row['menge_num']),
                    'po_qty': float(row['po_qty']),
                    'qty_variance': float(row['qty_variance']),
                    'qty_variance_pct': float(row['qty_variance_pct']),
                }
            )
            self.exceptions.append(exc)
    
    def _detect_three_way_mismatch(self, rbkp: pd.DataFrame, bseg: pd.DataFrame, 
                                    ekko: pd.DataFrame, ekpo: pd.DataFrame):
        """Detect 3-way match exceptions (PO -> GR -> Invoice)"""
        # This is a simplified version - full 3-way match needs MSEG (material docs)
        # For now, check if invoice exists without corresponding PO/GR
        
        if rbkp.empty or 'lifnr' not in rbkp.columns:
            return
        
        # Find invoices for vendors that have no POs
        if not ekko.empty:
            po_vendors = set(ekko['lifnr'].astype(str).str.strip().str.zfill(10).unique())
            rbkp_vendors = set(rbkp['lifnr'].astype(str).str.strip().str.zfill(10).unique())
            
            # Vendors with invoices but no POs (could be service invoices, expense, etc.)
            invoice_only_vendors = rbkp_vendors - po_vendors
            if invoice_only_vendors:
                sample = rbkp[rbkp['lifnr'].astype(str).str.strip().str.zfill(10).isin(invoice_only_vendors)].head(50)
                for _, row in sample.iterrows():
                    exc = InvoiceException(
                        exception_id=f"3WAY_NOPO_{row['belnr']}_{row['gjahr']}",
                        category=ExceptionCategory.THREE_WAY_MISMATCH,
                        severity=ExceptionSeverity.MEDIUM,
                        description=f"Invoice {row['belnr']}/{row['gjahr']} for vendor {row['lifnr']} has no matching PO",
                        affected_documents=[f"{row['belnr']}/{row['gjahr']}"],
                        vendor_id=str(row.get('lifnr', '')),
                        company_code=str(row.get('bukrs', '')),
                        amount=float(pd.to_numeric(row.get('rmwwr', 0), errors='coerce')),
                        currency=str(row.get('waers', '')),
                        root_cause="Invoice received for vendor without purchase order - could be services, expenses, or missing PO",
                        recommended_action="Verify if this is a non-PO invoice (services, utilities, expenses). "
                                          "If material/service should have PO, create PO retroactively or reject invoice.",
                        auto_resolvable=False,
                        confidence_score=0.60,
                        metadata={'type': 'no_po_for_invoice'}
                    )
                    self.exceptions.append(exc)
    
    def _detect_vendor_blocks(self, rbkp: pd.DataFrame):
        """Detect invoices for blocked vendors"""
        if rbkp.empty or self.lfa1.empty or 'lifnr' not in rbkp.columns:
            return
        
        blocked_vendors = self.lfa1[self.lfa1['sperr'] == 'X']['lifnr'].unique()
        if len(blocked_vendors) == 0:
            return
        
        rbkp['lifnr_clean'] = rbkp['lifnr'].astype(str).str.strip().str.zfill(10)
        blocked_invoices = rbkp[rbkp['lifnr_clean'].isin(blocked_vendors)]
        
        for _, row in blocked_invoices.head(100).iterrows():
            exc = InvoiceException(
                exception_id=f"VENDBLOCK_{row['belnr']}_{row['gjahr']}",
                category=ExceptionCategory.VENDOR_BLOCK,
                severity=ExceptionSeverity.HIGH,
                description=f"Invoice {row['belnr']}/{row['gjahr']} for blocked vendor {row['lifnr']}",
                affected_documents=[f"{row['belnr']}/{row['gjahr']}"],
                vendor_id=str(row.get('lifnr', '')),
                company_code=str(row.get('bukrs', '')),
                amount=float(pd.to_numeric(row.get('rmwwr', 0), errors='coerce')),
                currency=str(row.get('waers', '')),
                root_cause="Vendor master has payment block (SPERR=X) - vendor may be on hold, bankrupt, or under review",
                recommended_action="Check vendor master (FK03) for block reason. Resolve vendor issue before processing payment. "
                                  "Coordinate with procurement/finance on vendor status.",
                auto_resolvable=False,
                confidence_score=0.95,
                metadata={'vendor_block_reason': 'Master data block'}
            )
            self.exceptions.append(exc)
    
    def _detect_tax_issues(self, rbkp: pd.DataFrame, bseg: pd.DataFrame):
        """Detect tax calculation issues"""
        if rbkp.empty:
            return
        
        rbkp['wmwst1_num'] = pd.to_numeric(rbkp.get('wmwst1', 0), errors='coerce').fillna(0)
        rbkp['wmwst2_num'] = pd.to_numeric(rbkp.get('wmwst2', 0), errors='coerce').fillna(0)
        rbkp['rmwwr_num'] = pd.to_numeric(rbkp.get('rmwwr', 0), errors='coerce').fillna(0)
        
        # Invoices with tax but no net amount, or vice versa
        tax_issues = rbkp[
            ((rbkp['wmwst1_num'] > 0) | (rbkp['wmwst2_num'] > 0)) & 
            (rbkp['rmwwr_num'] == 0)
        ]
        
        for _, row in tax_issues.head(50).iterrows():
            exc = InvoiceException(
                exception_id=f"TAX_{row['belnr']}_{row['gjahr']}",
                category=ExceptionCategory.TAX_ISSUE,
                severity=ExceptionSeverity.MEDIUM,
                description=f"Invoice {row['belnr']}/{row['gjahr']} has tax amount but zero net amount",
                affected_documents=[f"{row['belnr']}/{row['gjahr']}"],
                vendor_id=str(row.get('lifnr', '')),
                company_code=str(row.get('bukrs', '')),
                amount=float(row['wmwst1_num'] + row['wmwst2_num']),
                currency=str(row.get('waers', '')),
                root_cause="Tax calculated on zero base amount - possible data entry error or tax-only invoice",
                recommended_action="Review invoice tax calculation. Check tax codes (MWSKZ) in line items. "
                                  "Verify if this is a valid tax-only transaction (e.g., import VAT).",
                auto_resolvable=False,
                confidence_score=0.70,
                metadata={
                    'tax_amount_1': float(row['wmwst1_num']),
                    'tax_amount_2': float(row['wmwst2_num']),
                    'tax_code_1': str(row.get('mwskz1', '')),
                    'tax_code_2': str(row.get('mwskz2', '')),
                }
            )
            self.exceptions.append(exc)
    
    def _detect_currency_mismatch(self, rbkp: pd.DataFrame, ekko: pd.DataFrame):
        """Detect currency mismatch between PO and Invoice"""
        if rbkp.empty or ekko.empty or 'lifnr' not in rbkp.columns:
            return
        
        # Get PO currencies per vendor
        po_curr = ekko.groupby('lifnr')['waers'].first().reset_index()
        po_curr.columns = ['lifnr', 'po_currency']
        po_curr['lifnr'] = po_curr['lifnr'].astype(str).str.strip().str.zfill(10)
        
        rbkp['lifnr_clean'] = rbkp['lifnr'].astype(str).str.strip().str.zfill(10)
        merged = rbkp.merge(po_curr, on='lifnr', how='inner')
        
        mismatch = merged[merged['waers'] != merged['po_currency']]
        
        for _, row in mismatch.head(50).iterrows():
            exc = InvoiceException(
                exception_id=f"CURRMIS_{row['belnr']}_{row['gjahr']}",
                category=ExceptionCategory.CURRENCY_MISMATCH,
                severity=ExceptionSeverity.MEDIUM,
                description=f"Currency mismatch: Invoice {row['waers']} vs PO {row['po_currency']} for vendor {row['lifnr']}",
                affected_documents=[f"{row['belnr']}/{row['gjahr']}"],
                vendor_id=str(row.get('lifnr', '')),
                company_code=str(row.get('bukrs', '')),
                amount=float(pd.to_numeric(row.get('rmwwr', 0), errors='coerce')),
                currency=str(row.get('waers', '')),
                root_cause="Invoice currency differs from PO currency - exchange rate risk, "
                          "possible wrong currency on invoice or PO",
                recommended_action="Verify correct currency with vendor. Check exchange rate in OB08. "
                                  "If PO currency wrong, change PO. If invoice currency wrong, request corrected invoice.",
                auto_resolvable=False,
                confidence_score=0.75,
                metadata={
                    'invoice_currency': str(row.get('waers', '')),
                    'po_currency': str(row.get('po_currency', '')),
                    'exchange_rate': float(row.get('kursf', 1)) if pd.notna(row.get('kursf')) else 1.0,
                }
            )
            self.exceptions.append(exc)
    
    def _detect_missing_master_data(self, rbkp: pd.DataFrame):
        """Detect invoices for vendors missing from master data"""
        if rbkp.empty or self.lfa1.empty:
            return
        
        rbkp['lifnr_clean'] = rbkp['lifnr'].astype(str).str.strip().str.zfill(10)
        master_vendors = set(self.lfa1['lifnr'].unique())
        invoice_vendors = set(rbkp['lifnr_clean'].unique())
        
        missing = invoice_vendors - master_vendors
        
        if missing:
            missing_invoices = rbkp[rbkp['lifnr_clean'].isin(missing)]
            for _, row in missing_invoices.head(50).iterrows():
                exc = InvoiceException(
                    exception_id=f"MISSMAST_{row['belnr']}_{row['gjahr']}",
                    category=ExceptionCategory.MISSING_MASTER_DATA,
                    severity=ExceptionSeverity.HIGH,
                    description=f"Invoice {row['belnr']}/{row['gjahr']} for vendor {row['lifnr']} not in vendor master",
                    affected_documents=[f"{row['belnr']}/{row['gjahr']}"],
                    vendor_id=str(row.get('lifnr', '')),
                    company_code=str(row.get('bukrs', '')),
                    amount=float(pd.to_numeric(row.get('rmwwr', 0), errors='coerce')),
                    currency=str(row.get('waers', '')),
                    root_cause="Vendor master record missing - vendor not created in system or deleted",
                    recommended_action="Create vendor master (FK01) with correct payment terms, bank details, tax info. "
                                      "Or verify if vendor ID is correct on invoice.",
                    auto_resolvable=False,
                    confidence_score=0.90,
                    metadata={}
                )
                self.exceptions.append(exc)

class ExceptionResolutionAgent:
    """Agent that provides resolution workflows for exceptions"""
    
    def __init__(self, detector: InvoiceExceptionDetector):
        self.detector = detector
        self.resolution_strategies = self._init_strategies()
    
    def _init_strategies(self) -> Dict[ExceptionCategory, Dict]:
        return {
            ExceptionCategory.PAYMENT_BLOCK: {
                'steps': [
                    "Check invoice line items for errors (price, qty, tax)",
                    "Verify goods receipt posted (MIGO)",
                    "Check vendor master for payment blocks (FK03)",
                    "Review tolerance limits (OLMx)",
                    "Release block via MRBR if all checks pass",
                    "If unresolved, escalate to AP supervisor"
                ],
                'tcode': 'MRBR',
                'auto_releasable': False
            },
            ExceptionCategory.PRICE_VARIANCE: {
                'steps': [
                    "Compare invoice price to PO price and info record",
                    "Check for approved price change (ME11/ME12)",
                    "Verify quantity discounts or surcharges",
                    "Check for freight/handling charges not on PO",
                    "If variance within tolerance, auto-accept",
                    "If outside tolerance, request credit note or PO change"
                ],
                'tcode': 'MIRO',
                'auto_releasable': True,
                'tolerance_check': 'OLMx'
            },
            ExceptionCategory.QUANTITY_VARIANCE: {
                'steps': [
                    "Check goods receipt quantity (MIGO/MB51)",
                    "Verify actual received quantity with warehouse",
                    "Check for partial deliveries",
                    "If GR matches invoice, variance is PO vs GR issue",
                    "Request credit/debit note for difference"
                ],
                'tcode': 'MIGO',
                'auto_releasable': False
            },
            ExceptionCategory.THREE_WAY_MISMATCH: {
                'steps': [
                    "Identify missing document (PO, GR, or Invoice)",
                    "For non-PO invoices: verify expense category",
                    "For missing GR: check warehouse receipt",
                    "For missing PO: create PO or reject invoice"
                ],
                'tcode': 'MIRO',
                'auto_releasable': False
            },
            ExceptionCategory.OVERDUE: {
                'steps': [
                    "Run dunning proposal (F150)",
                    "Contact vendor for payment status",
                    "Check for disputes or quality holds",
                    "Escalate to vendor manager if >90 days",
                    "Consider payment hold or legal action"
                ],
                'tcode': 'F150',
                'auto_releasable': True
            },
            ExceptionCategory.DUPLICATE: {
                'steps': [
                    "Identify original vs duplicate invoice",
                    "Verify with vendor which is correct",
                    "Park duplicate for review (MIR7)",
                    "Cancel/credit duplicate invoice",
                    "Update vendor master to prevent future duplicates"
                ],
                'tcode': 'MIR7',
                'auto_releasable': False
            },
            ExceptionCategory.VENDOR_BLOCK: {
                'steps': [
                    "Check vendor master block reason (FK03)",
                    "Contact procurement for vendor status",
                    "Resolve underlying issue (quality, legal, financial)",
                    "Remove block if vendor cleared"
                ],
                'tcode': 'FK03',
                'auto_releasable': False
            },
            ExceptionCategory.MISSING_MASTER_DATA: {
                'steps': [
                    "Verify vendor details from invoice",
                    "Create vendor master (FK01) with correct data",
                    "Set up payment terms, bank, tax info",
                    "Re-process invoice"
                ],
                'tcode': 'FK01',
                'auto_releasable': True
            },
            ExceptionCategory.TAX_ISSUE: {
                'steps': [
                    "Review tax codes on invoice lines (MWSKZ)",
                    "Check tax calculation logic (FTXP)",
                    "Verify tax jurisdiction and rates",
                    "Correct tax code or amount"
                ],
                'tcode': 'MIRO',
                'auto_releasable': True
            },
            ExceptionCategory.CURRENCY_MISMATCH: {
                'steps': [
                    "Confirm correct currency with vendor",
                    "Check exchange rate in OB08",
                    "Update PO or request corrected invoice",
                    "Re-calculate with correct rate"
                ],
                'tcode': 'OB08',
                'auto_releasable': False
            },
        }
    
    def generate_resolution_plan(self, exception: InvoiceException) -> Dict:
        """Generate detailed resolution plan for an exception"""
        strategy = self.resolution_strategies.get(exception.category, {})
        
        return {
            'exception_id': exception.exception_id,
            'category': exception.category.value,
            'severity': exception.severity.value,
            'summary': exception.description,
            'root_cause': exception.root_cause,
            'recommended_action': exception.recommended_action,
            'auto_resolvable': exception.auto_resolvable,
            'confidence': exception.confidence_score,
            'resolution_steps': strategy.get('steps', ['Manual investigation required']),
            'primary_tcode': strategy.get('tcode', 'MIRO'),
            'affected_documents': exception.affected_documents,
            'vendor': exception.vendor_id,
            'amount': exception.amount,
            'currency': exception.currency,
            'metadata': exception.metadata
        }
    
    def process_batch(self, exceptions: List[InvoiceException]) -> List[Dict]:
        """Process a batch of exceptions and generate resolution plans"""
        plans = []
        for exc in exceptions:
            plan = self.generate_resolution_plan(exc)
            plans.append(plan)
        return plans
    
    def get_priority_queue(self, exceptions: List[InvoiceException]) -> List[InvoiceException]:
        """Sort exceptions by priority (severity + amount + age)"""
        severity_order = {
            ExceptionSeverity.CRITICAL: 0,
            ExceptionSeverity.HIGH: 1,
            ExceptionSeverity.MEDIUM: 2,
            ExceptionSeverity.LOW: 3,
            ExceptionSeverity.INFO: 4
        }
        
        return sorted(exceptions, key=lambda e: (
            severity_order.get(e.severity, 5),
            -e.amount,
            -e.confidence_score
        ))

class ExceptionReporter:
    """Generate reports and dashboards for exceptions"""
    
    @staticmethod
    def to_dataframe(exceptions: List[InvoiceException]) -> pd.DataFrame:
        data = []
        for exc in exceptions:
            data.append({
                'exception_id': exc.exception_id,
                'category': exc.category.value,
                'severity': exc.severity.value,
                'description': exc.description,
                'vendor_id': exc.vendor_id,
                'company_code': exc.company_code,
                'amount': exc.amount,
                'currency': exc.currency,
                'root_cause': exc.root_cause,
                'recommended_action': exc.recommended_action,
                'auto_resolvable': exc.auto_resolvable,
                'confidence_score': exc.confidence_score,
                'detected_at': exc.detected_at,
                'affected_docs': '; '.join(exc.affected_documents),
            })
        return pd.DataFrame(data)
    
    @staticmethod
    def summary_by_category(exceptions: List[InvoiceException]) -> pd.DataFrame:
        df = ExceptionReporter.to_dataframe(exceptions)
        if df.empty:
            return pd.DataFrame()
        
        summary = df.groupby(['category', 'severity']).agg(
            count=('exception_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_confidence=('confidence_score', 'mean')
        ).reset_index()
        return summary.sort_values(['severity', 'count'], ascending=[True, False])
    
    @staticmethod
    def summary_by_vendor(exceptions: List[InvoiceException], top_n: int = 20) -> pd.DataFrame:
        df = ExceptionReporter.to_dataframe(exceptions)
        if df.empty:
            return pd.DataFrame()
        
        summary = df.groupby('vendor_id').agg(
            exception_count=('exception_id', 'count'),
            total_amount=('amount', 'sum'),
            categories=('category', lambda x: list(x.unique())),
            max_severity=('severity', lambda x: min(x, key=lambda s: ['CRITICAL','HIGH','MEDIUM','LOW','INFO'].index(s)))
        ).reset_index()
        return summary.sort_values('exception_count', ascending=False).head(top_n)
    
    @staticmethod
    def export_json(exceptions: List[InvoiceException], filepath: str):
        """Export exceptions to JSON for API integration"""
        data = []
        for exc in exceptions:
            data.append({
                'exception_id': exc.exception_id,
                'category': exc.category.value,
                'severity': exc.severity.value,
                'description': exc.description,
                'affected_documents': exc.affected_documents,
                'vendor_id': exc.vendor_id,
                'company_code': exc.company_code,
                'amount': exc.amount,
                'currency': exc.currency,
                'root_cause': exc.root_cause,
                'recommended_action': exc.recommended_action,
                'auto_resolvable': exc.auto_resolvable,
                'confidence_score': exc.confidence_score,
                'metadata': exc.metadata,
                'detected_at': exc.detected_at.isoformat()
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data)} exceptions to {filepath}")

def main():
    print("=" * 60)
    print("INVOICE EXCEPTION RESOLUTION AGENT")
    print("=" * 60)
    
    # Initialize
    loader = DataLoader(DATA_DIR)
    detector = InvoiceExceptionDetector(loader)
    agent = ExceptionResolutionAgent(detector)
    reporter = ExceptionReporter()
    
    # Detect exceptions
    exceptions = detector.detect_all(sample_size=100000)
    
    # Generate reports
    print("\n" + "=" * 60)
    print("EXCEPTION SUMMARY BY CATEGORY")
    print("=" * 60)
    cat_summary = reporter.summary_by_category(exceptions)
    print(cat_summary.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TOP VENDORS WITH EXCEPTIONS")
    print("=" * 60)
    vendor_summary = reporter.summary_by_vendor(exceptions, top_n=15)
    print(vendor_summary.to_string(index=False))
    
    # Priority queue
    print("\n" + "=" * 60)
    print("TOP 10 PRIORITY EXCEPTIONS")
    print("=" * 60)
    priority = agent.get_priority_queue(exceptions)
    for i, exc in enumerate(priority[:10], 1):
        print(f"\n{i}. [{exc.severity.value}] {exc.exception_id}")
        print(f"   Vendor: {exc.vendor_id} | Amount: {exc.amount:,.2f} {exc.currency}")
        print(f"   {exc.description}")
        print(f"   Action: {exc.recommended_action[:100]}...")
    
    # Export for integration
    reporter.export_json(exceptions, '/home/noirxvii/exceptions_output.json')
    
    # Export detailed CSV
    df = reporter.to_dataframe(exceptions)
    df.to_csv('/home/noirxvii/exceptions_detailed.csv', index=False)
    print(f"\nDetailed exceptions saved to /home/noirxvii/exceptions_detailed.csv")
    
    # Show resolution plan for top exception
    if priority:
        print("\n" + "=" * 60)
        print("RESOLUTION PLAN FOR TOP PRIORITY EXCEPTION")
        print("=" * 60)
        plan = agent.generate_resolution_plan(priority[0])
        for key, value in plan.items():
            if key != 'resolution_steps':
                print(f"  {key}: {value}")
        print("  resolution_steps:")
        for i, step in enumerate(plan['resolution_steps'], 1):
            print(f"    {i}. {step}")
    
    return exceptions, agent, reporter

if __name__ == "__main__":
    exceptions, agent, reporter = main()