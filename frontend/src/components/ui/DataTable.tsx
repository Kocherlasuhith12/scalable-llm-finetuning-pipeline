"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Download,
  Filter,
  ChevronLeft,
  ChevronRight,
  MoreVertical,
  Layers,
  Inbox
} from "lucide-react";
import { Button } from "./Button";
import { Input } from "./Input";
import { Badge } from "./Badge";
import { Dropdown, DropdownItem, DropdownDivider } from "./Dropdown";
import { SkeletonRow } from "./Skeleton";

export interface Column<T> {
  key: string;
  label: string;
  sortable?: boolean;
  align?: "left" | "center" | "right";
  render?: (item: T, index: number) => React.ReactNode;
}

export interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  title?: string;
  description?: string;
  searchable?: boolean;
  searchPlaceholder?: string;
  searchKeys?: (keyof T)[];
  filterable?: boolean;
  filterKey?: keyof T;
  filterOptions?: { label: string; value: string }[];
  exportable?: boolean;
  exportFileName?: string;
  pageSize?: number;
  isLoading?: boolean;
  emptyTitle?: string;
  emptyDescription?: string;
  onRowClick?: (item: T) => void;
  rowActions?: (item: T) => { label: string; icon?: React.ReactNode; danger?: boolean; onClick: () => void }[];
}

export function DataTable<T extends Record<string, any>>({
  columns,
  data,
  title,
  description,
  searchable = true,
  searchPlaceholder = "Search records...",
  searchKeys,
  filterable = false,
  filterKey,
  filterOptions = [],
  exportable = true,
  exportFileName = "export",
  pageSize = 5,
  isLoading = false,
  emptyTitle = "No records found",
  emptyDescription = "There are no matching data records to display.",
  onRowClick,
  rowActions
}: DataTableProps<T>) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFilter, setSelectedFilter] = useState("ALL");
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc" | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(pageSize);

  // 1. Search & Filter
  const filteredData = useMemo(() => {
    return data.filter((item) => {
      if (filterable && filterKey && selectedFilter !== "ALL") {
        const itemVal = String(item[filterKey] || "").toLowerCase();
        if (itemVal !== selectedFilter.toLowerCase()) return false;
      }

      if (!searchQuery.trim()) return true;

      const q = searchQuery.toLowerCase();
      if (searchKeys && searchKeys.length > 0) {
        return searchKeys.some((k) => String(item[k] || "").toLowerCase().includes(q));
      }

      return Object.values(item).some((val) => String(val || "").toLowerCase().includes(q));
    });
  }, [data, searchQuery, selectedFilter, filterable, filterKey, searchKeys]);

  // 2. Sorting
  const sortedData = useMemo(() => {
    if (!sortKey || !sortDirection) return filteredData;

    return [...filteredData].sort((a, b) => {
      const valA = a[sortKey];
      const valB = b[sortKey];

      if (valA === valB) return 0;
      if (valA === null || valA === undefined) return 1;
      if (valB === null || valB === undefined) return -1;

      const comp = typeof valA === "number" && typeof valB === "number"
        ? valA - valB
        : String(valA).localeCompare(String(valB));

      return sortDirection === "asc" ? comp : -comp;
    });
  }, [filteredData, sortKey, sortDirection]);

  // 3. Pagination
  const totalPages = Math.ceil(sortedData.length / rowsPerPage) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * rowsPerPage;
    return sortedData.slice(start, start + rowsPerPage);
  }, [sortedData, currentPage, rowsPerPage]);

  const handleSort = (key: string) => {
    if (sortKey !== key) {
      setSortKey(key);
      setSortDirection("asc");
    } else if (sortDirection === "asc") {
      setSortDirection("desc");
    } else {
      setSortKey(null);
      setSortDirection(null);
    }
  };

  // 4. CSV Export
  const handleExportCSV = () => {
    if (sortedData.length === 0) return;

    const headers = columns.map((c) => c.label).join(",");
    const rows = sortedData.map((item) =>
      columns.map((c) => {
        let val = item[c.key];
        if (typeof val === "object") val = JSON.stringify(val);
        return `"${String(val ?? "").replace(/"/g, '""')}"`;
      }).join(",")
    );

    const csvContent = "data:text/csv;charset=utf-8," + [headers, ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${exportFileName}_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-[#171A21]/90 backdrop-blur-md border border-[#27272A] rounded-xl overflow-hidden shadow-lg space-y-0">
      {/* Header bar */}
      <div className="p-4 border-b border-[#27272A] bg-[#111318]/80 flex flex-wrap items-center justify-between gap-3">
        <div>
          {title && <h3 className="text-sm font-bold text-[#F8FAFC] tracking-tight">{title}</h3>}
          {description && <p className="text-xs text-[#94A3B8] mt-0.5">{description}</p>}
        </div>

        <div className="flex items-center gap-2.5 flex-wrap">
          {searchable && (
            <div className="w-56">
              <Input
                leftIcon={<Search className="w-3.5 h-3.5" />}
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setCurrentPage(1);
                }}
                placeholder={searchPlaceholder}
                className="mb-0 py-1.5"
              />
            </div>
          )}

          {filterable && filterOptions.length > 0 && (
            <div className="flex items-center gap-1 bg-[#09090B] border border-[#27272A] rounded-lg p-1">
              <button
                onClick={() => setSelectedFilter("ALL")}
                className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
                  selectedFilter === "ALL" ? "bg-[#8B5CF6] text-white" : "text-[#94A3B8] hover:text-[#F8FAFC]"
                }`}
              >
                All
              </button>
              {filterOptions.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setSelectedFilter(opt.value)}
                  className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
                    selectedFilter === opt.value ? "bg-[#8B5CF6] text-white" : "text-[#94A3B8] hover:text-[#F8FAFC]"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}

          {exportable && (
            <Button
              variant="secondary"
              size="sm"
              onClick={handleExportCSV}
              leftIcon={<Download className="w-3.5 h-3.5" />}
            >
              Export CSV
            </Button>
          )}
        </div>
      </div>

      {/* Table Area */}
      <div className="w-full overflow-x-auto max-h-[480px]">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-[#111318]/95 backdrop-blur-md border-b border-[#27272A] text-[#64748B] uppercase tracking-wider text-[10px] font-bold z-10">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => col.sortable !== false && handleSort(col.key)}
                  className={`py-3.5 px-4 select-none ${
                    col.sortable !== false ? "cursor-pointer hover:text-[#F8FAFC]" : ""
                  } ${col.align === "right" ? "text-right" : col.align === "center" ? "text-center" : "text-left"}`}
                >
                  <div className="inline-flex items-center gap-1.5">
                    <span>{col.label}</span>
                    {col.sortable !== false && (
                      <span className="text-[#8B5CF6]">
                        {sortKey === col.key ? (
                          sortDirection === "asc" ? (
                            <ArrowUp className="w-3 h-3" />
                          ) : (
                            <ArrowDown className="w-3 h-3" />
                          )
                        ) : (
                          <ArrowUpDown className="w-3 h-3 opacity-30" />
                        )}
                      </span>
                    )}
                  </div>
                </th>
              ))}
              {rowActions && <th className="py-3.5 px-4 text-right">Actions</th>}
            </tr>
          </thead>

          <tbody className="divide-y divide-[#27272A]/40 text-[#F8FAFC]">
            {isLoading ? (
              Array.from({ length: rowsPerPage }).map((_, i) => (
                <SkeletonRow key={i} columns={columns.length + (rowActions ? 1 : 0)} />
              ))
            ) : paginatedData.length === 0 ? (
              <tr>
                <td colSpan={columns.length + (rowActions ? 1 : 0)} className="py-12 text-center">
                  <div className="max-w-xs mx-auto space-y-2">
                    <div className="w-10 h-10 rounded-full bg-[#8B5CF6]/15 text-[#A78BFA] flex items-center justify-center mx-auto">
                      <Inbox className="w-5 h-5" />
                    </div>
                    <div className="font-semibold text-sm text-[#F8FAFC]">{emptyTitle}</div>
                    <div className="text-xs text-[#94A3B8]">{emptyDescription}</div>
                  </div>
                </td>
              </tr>
            ) : (
              paginatedData.map((item, idx) => {
                const actionsList = rowActions ? rowActions(item) : [];
                return (
                  <motion.tr
                    key={item.id || idx}
                    whileHover={{ backgroundColor: "rgba(31, 37, 48, 0.6)" }}
                    transition={{ duration: 0.15 }}
                    onClick={() => onRowClick && onRowClick(item)}
                    className={`transition-colors ${onRowClick ? "cursor-pointer" : ""}`}
                  >
                    {columns.map((col) => (
                      <td
                        key={col.key}
                        className={`py-3.5 px-4 align-middle ${
                          col.align === "right" ? "text-right" : col.align === "center" ? "text-center" : "text-left"
                        }`}
                      >
                        {col.render ? col.render(item, idx) : item[col.key]}
                      </td>
                    ))}

                    {rowActions && (
                      <td className="py-3.5 px-4 text-right align-middle" onClick={(e) => e.stopPropagation()}>
                        <Dropdown
                          align="right"
                          trigger={
                            <button className="p-1 rounded-lg text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-[#111318] transition-all cursor-pointer">
                              <MoreVertical className="w-4 h-4" />
                            </button>
                          }
                        >
                          {actionsList.map((action, aIdx) => (
                            <DropdownItem
                              key={aIdx}
                              danger={action.danger}
                              icon={action.icon}
                              onClick={action.onClick}
                            >
                              {action.label}
                            </DropdownItem>
                          ))}
                        </Dropdown>
                      </td>
                    )}
                  </motion.tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="p-3.5 border-t border-[#27272A] bg-[#111318]/90 flex items-center justify-between text-xs text-[#94A3B8]">
        <div className="flex items-center gap-3">
          <span>
            Showing <strong className="text-[#F8FAFC]">{sortedData.length > 0 ? (currentPage - 1) * rowsPerPage + 1 : 0}</strong> to{" "}
            <strong className="text-[#F8FAFC]">{Math.min(currentPage * rowsPerPage, sortedData.length)}</strong> of{" "}
            <strong className="text-[#F8FAFC]">{sortedData.length}</strong> records
          </span>

          <select
            value={rowsPerPage}
            onChange={(e) => {
              setRowsPerPage(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="bg-[#09090B] border border-[#27272A] rounded px-2 py-1 text-xs text-[#F8FAFC] focus:outline-none cursor-pointer"
          >
            <option value={5}>5 per page</option>
            <option value={10}>10 per page</option>
            <option value={25}>25 per page</option>
          </select>
        </div>

        <div className="flex items-center gap-1">
          <Button
            variant="secondary"
            size="sm"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
            leftIcon={<ChevronLeft className="w-3.5 h-3.5" />}
          />

          <span className="px-3 py-1 font-semibold text-[#F8FAFC]">
            Page {currentPage} of {totalPages}
          </span>

          <Button
            variant="secondary"
            size="sm"
            disabled={currentPage >= totalPages}
            onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
            leftIcon={<ChevronRight className="w-3.5 h-3.5" />}
          />
        </div>
      </div>
    </div>
  );
}
