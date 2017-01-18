require.config({
  paths: {
    "d3": "http://d3js.org/d3.v3.min"
  }
});

require(['d3', 'jquery'], function (d3, $) {
  //{% include 'sankey.js' %}//

  console.log('defining function');

  !function (csv, container, css) {
    'use strict';

    function uniqueID() {
      function chr4() {
        return Math.random().toString(16).slice(-4);
      }

      function chr9() {
        return Math.random().toString(36).substr(2, 9);
      }

      return '_' + chr9();
    }

    var id = uniqueID();

    console.log('id', id);

    var div = $('<div id="' + id + '"></div>').appendTo(container);

    var units = "Annotation";

    var margin = {top: 10, right: 10, bottom: 10, left: 10},
        width  = 800 - margin.left - margin.right,
        height = 600 - margin.top - margin.bottom;

    var formatNumber = d3.format(",.0f"),    // zero decimal places
        format       = function (d) {
          return formatNumber(d) + " " + units + (d > 1 ? "s" : "");
        },
        color        = d3.scale.category20();

    // console.log('adding svg elt.');
    $('<style>' + css + '</style>').appendTo(div);

    // append the svg canvas to the page
    var svg = d3.select('#' + id).append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    // Set the sankey diagram properties
    var sankey = d3.sankey()
      .nodeWidth(12)
      .nodePadding(10)
      .size([width, height]);

    var path = sankey.link();

    var Node = function(id, fe, gf, core) {
      return { name: id, FE: fe, GF: gf, core: core === "True" };
    };

    var eq = function (n1, n2) { return n1.name === n2.name; };

    // load the data (using the timely portfolio csv method)
    var data = d3.csv.parse(csv);

    //set up graph in same style as original example but empty
    var graph = {"nodes": [], "links": []};

    graph.nodes = d3.merge(data.map(function (d) {
        return [Node(d.source_id, d.source_FE, d.source_GF, d.source_core),
                Node(d.target_id, d.target_FE, d.target_GF, d.target_core)];
      }))
      .sort(function (n1, n2) { return d3.ascending(n1.name, n2.name); })
      .reduceRight(function (ns, n) {
        if (ns.length > 0 && eq(n, ns[0])) return ns;
        else                               return [n].concat(ns);
      }, []);

    var nodeMap = d3.map(graph.nodes, function (n) { return n.name; });

    // console.log('nodes:', graph.nodes);
    // console.log('nodeMap:', nodeMap);

    //   var totalLen = d3.sum(data, function (d) {
    //     return +d.value; })

    graph.links = data.map(function (d) {
      return {
        "source": nodeMap.get(d.source_id),
        "target": nodeMap.get(d.target_id),
        "value":  +d.count
      };
    });

    // console.log('links:', graph.links);

    sankey
      .nodes(graph.nodes)
      .links(graph.links)
      .layout(32);

    // add in the links
    var link = svg.append("g").selectAll(".link")
      .data(graph.links)
      .enter().append("path")
      .attr("class", "link")
      .attr("d", path)
      .style("stroke-width", function (d) { return Math.max(1, d.dy); })
      .sort(function (a, b) { return b.dy - a.dy; });

    // add the link titles
    link.append("title")
      .text(function (d) {
        var s = d.source,
            t = d.target;
        return  s.GF  + ": " + s.FE + " (core: " + s.core + ") ⟶ "
              + t.GF  + ": " + t.FE + " (core: " + t.core + ")\n"
               + format(d.value);
      });

    // add in the nodes
    var node = svg.append("g").selectAll(".node")
      .data(graph.nodes)
      .enter().append("g")
      .attr("class", "node")
      .attr("transform", function (d) { return "translate(" + d.x + "," + d.y + ")"; })
      .call(d3.behavior.drag()
              .origin(function (d) { return d; })
              .on("dragstart", function () { this.parentNode.appendChild(this); })
              .on("drag", dragmove));

    // add the rectangles for the nodes
    node.append("rect")
      .attr("height", function (d) { return d.dy; })
      .attr("width", sankey.nodeWidth())
      .style("fill", function (d) { return d.color = color(d.name.replace(/ .*/, "")); })
      .style("stroke", function (d) { return d3.rgb(d.color).darker(2); })
      .append("title")
      .text(function(d) { return d.GF + ': ' + d.FE + " (core: " + d.core + ")\n" + format(d.value); });

    // add in the title for the nodes
    node.append("text")
      .attr("x", -6)
      .attr("y", function (d) { return d.dy / 2; })
      .attr("dy", ".35em")
      .attr("text-anchor", "end")
      .attr("transform", null)
      .text(function (d) { return d.GF + ': ' + d.FE; })
      .filter(function (d) { return d.x < width / 2; })
      .attr("x", 6 + sankey.nodeWidth())
      .attr("text-anchor", "start");

    // the function for moving the nodes
    function dragmove(d) {
      d3.select(this)
        .attr("transform", "translate("
                           + (d.x = Math.max(0, Math.min(width - d.dx, d3.event.x)))
                           + ","
                           + (d.y = Math.max(0, Math.min(height - d.dy, d3.event.y)))
                           + ")");
      sankey.relayout();
      link.attr("d", path);
    }
  }(/*{{ arguments }}*/)
});
