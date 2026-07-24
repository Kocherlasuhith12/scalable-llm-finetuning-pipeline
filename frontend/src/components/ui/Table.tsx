import React from "react";

export const Table: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <div className="w-full overflow-x-auto">
    <table className={`w-full text-left text-xs ${className}`}>{children}</table>
  </div>
);

export const TableHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <thead className={`border-b border-[#27272A] text-[#64748B] uppercase tracking-wider text-[10px] font-semibold ${className}`}>
    {children}
  </thead>
);

export const TableBody: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <tbody className={`divide-y divide-[#27272A]/50 text-[#F8FAFC] ${className}`}>{children}</tbody>
);

export const TableRow: React.FC<{ children: React.ReactNode; className?: string; onClick?: () => void }> = ({
  children,
  className = "",
  onClick
}) => (
  <tr
    onClick={onClick}
    className={`hover:bg-[#1F2530]/50 transition-colors ${onClick ? "cursor-pointer" : ""} ${className}`}
  >
    {children}
  </tr>
);

export const TableHead: React.FC<{ children?: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <th className={`pb-3 pt-2 px-3 font-semibold ${className}`}>{children}</th>
);

export const TableCell: React.FC<{ children?: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <td className={`py-3 px-3 align-middle ${className}`}>{children}</td>
);

export const TableEmpty: React.FC<{ colSpan: number; message?: string }> = ({
  colSpan,
  message = "No data available."
}) => (
  <tr>
    <td colSpan={colSpan} className="py-8 text-center text-xs text-[#64748B]">
      {message}
    </td>
  </tr>
);
