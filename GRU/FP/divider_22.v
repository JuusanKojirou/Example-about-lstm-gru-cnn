

module divider_22(rst_n,clock,start_sig,done_sig,a,b,c,d);

input rst_n,clock,start_sig;
output done_sig;
input [21:0] a;
input [7:0] b;
output [15:0] c,d;

reg [15:0] c,d;
reg [21:0] a_temp,b_temp,c_temp;
reg [41:0] c_total;
reg symbol,done_sig;
reg [3:0] i;
reg [5:0] bits;

always@(posedge clock or negedge rst_n)
begin
   if(!rst_n)
   begin
    c<=16'b0;
    d<=16'b0;
      i<=4'b0;
    done_sig<=1'b0;
    bits <= 6'b0;
    end
   else if(start_sig)
    case(i)  
    0:begin
          symbol<=a[21]^b[7];
		 a_temp <= a[21]?(~a+22'b1):a;
       b_temp<={b[7]?(~b+8'b1):b,14'b0};
       c_temp <=22'b0;
       c <=16'b0;
		 d <= 16'b0;
       c_total<=0;
       bits <= 5'b0; 
  //     d_temp <=32'b0;
          i<=i+4'b1;
        end
      1:begin
        if(b_temp==22'b0) begin d<=a_temp;c_temp<=symbol?(~c_temp+16'b1):c_temp;i<=i+4'b1;end
          else if(a_temp<b_temp) begin b_temp<=b_temp>>>1;c_temp<=c_temp<<<1;bits <= bits+6'b1;end
          else begin a_temp<=a_temp-b_temp;c_temp<=c_temp+16'b1; end
        end
      2:begin
      c_total<={c_temp,20'b0};
      i<=i+4'b1;
      end
    3:begin
      if(bits>0)begin
       c_total<=c_total>>1;
       bits<=bits-6'd1;
          end
       else
       i<=i+4'b1;
        end 
      4:begin
       c<=c_total[21:6];
       done_sig<=1'b1;
        end
    endcase
  end 
endmodule 
    
  

    